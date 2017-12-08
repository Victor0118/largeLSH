package largelsh

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{row_number, _}
import org.apache.spark.{SparkConf, SparkContext}


object LSH_kNN {

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1000000.0 + "ms")
    result
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("LSH at Large").setMaster("local")
    val sc = new SparkContext(sparkConf)
    sparkConf.set("spark.driver.allowMultipleContexts", "true")

    val spark: SparkSession = SparkSession.builder().appName("LSH at Large").config("spark.master", "local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // load mnist dataset using mllib library
    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    val trainingNumFeats = training.take(1)(0).features.size

    // change RDD type with mllib Vector to DataFrame type with ml Vector
    val training_df = training.toDF()
    val training_df_ml = MLUtils.convertVectorColumnsToML(training_df)
    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")
    val testingNumFeats = testing.take(1)(0).features.size

    val testing_padded = testing.map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(trainingNumFeats, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    })

    val testing_df = testing_padded.toDF()
    val testing_df_ml = MLUtils.convertVectorColumnsToML(testing_df)
    training_df_ml.select("features").show()
    testing_df_ml.select("features").show()

    val df_sample = testing_df_ml.sample(false, 1)

    val threshold = 2000

    val bl = 2.0
    val nht = 2
    val k = 5
    var brp = new BucketedRandomProjectionLSH().setBucketLength(bl).setNumHashTables(nht).setInputCol("features").setOutputCol("hashes")
    var model = brp.fit(training_df_ml)
    var transformedA = model.transform(training_df_ml)
    for (bl <- List(2.0, 5.0, 8.0)) {
      for (nht <- List(3, 5, 7)) {
        time {
          println("==========transform training data==========")
          brp = new BucketedRandomProjectionLSH().setBucketLength(bl).setNumHashTables(nht).setInputCol("features").setOutputCol("hashes")
          model = brp.fit(training_df_ml)
          transformedA = model.transform(training_df_ml)
        }
        for (k <- List(1, 5, 9)) {
          time {
            println("==========run kNN on testing data==========")
            // Compute the locality sensitive hashes for the input rows, then perform approximate similarity join.
            val results = model.approxSimilarityJoin(transformedA, df_sample, threshold, "EuclideanDistance")

            results.createOrReplaceTempView("results")
            val sqlDF = spark.sql("SELECT datasetA.label as label_train, datasetB.label as label_test, datasetB.features, EuclideanDistance FROM results ")

            // choose the majority from first k candidates
            val w = Window.partitionBy($"features").orderBy($"EuclideanDistance".asc)
            val dfTopk = sqlDF.withColumn("rn", row_number.over(w)).where($"rn" <= k).drop("rn")

            val finalDF = dfTopk.map(t => {
              val acc = if (t.getAs(0) == t.getAs(1)) 1 else 0
              (acc: Int, t.getAs(1): Double, t.getAs(2): org.apache.spark.ml.linalg.SparseVector, t.getAs(3): Double)
            })

            // group by testing samples and compute the accuracy
            val scores = finalDF.groupBy("_3").agg(sum("_1") / count("_1"))
            val accuracy = scores.map(t => if (t.getDouble(1) > 0.5) 1 else 0).reduce { (x, y) => x + y }.toDouble / df_sample.count()
            println("bl:", bl, "nht:", nht, "k:", k, "accuracy:", accuracy)
          }

        }
      }
    }
  }
}