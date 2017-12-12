package largelsh

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{row_number, _}
import org.rogach.scallop._

class SparkLSHConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val bl = opt[Double](default = Some(2.0), descr = "Bucket length")
  val nht = opt[Int](default = Some(3), descr = "Number of hash tables")
  val k = opt[Int](default = Some(1), descr = "Number of nearest neighbor in k-NN")
  val sample = opt[Int](default = None, descr = "Run on sample")
  val dataset = opt[String](default = Some("mnist"), descr = "Dataset to run on, mnist or svhn")
  val mode = opt[String](default = Some("eval"), descr = "Use eval to run on parameters provided, search to search over space of parameters")
  verify()
}


object SparkLSH {

  def main(args: Array[String]) {
    val conf = new SparkLSHConf(args)
    val spark: SparkSession = SparkSession.builder().appName("LargeLSH").getOrCreate()
    val sc = spark.sparkContext
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    val (training, testing) = DataLoader.getDatasets(conf.dataset(), conf.sample.toOption, sc)

    val trainingNumFeats = training.take(1)(0).features.size
    // change RDD type with mllib Vector to DataFrame type with ml Vector
    val training_df = training.toDF()
    val training_df_ml = MLUtils.convertVectorColumnsToML(training_df)
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

    /**
      * threshold: max l2 distance to filter before sorting
      * bl: BucketLength, W in LSH at large paper
      * nht: number of HashTables
      * k: number of nearest neighbor in k-NN
      */

    val threshold = trainingNumFeats * 2.5
    val bl = conf.bl()
    val nht = conf.nht()
    val k = conf.k()
    var brp = new BucketedRandomProjectionLSH()
                   .setBucketLength(bl)
                   .setNumHashTables(nht)
                   .setInputCol("features")
                   .setOutputCol("hashes")

    var model = brp.fit(training_df_ml)
    var transformedA = model.transform(training_df_ml)

    val searchMode = conf.mode() == "search"
    val blSpace = if (searchMode) Seq(2.0, 5.0, 8.0) else Seq(conf.bl())
    val nhtSpace = if (searchMode) Seq(3, 5, 7) else Seq(conf.nht())
    val kSpace = if (searchMode) Seq(1, 5, 9) else Seq(conf.k())

    for (bl <- blSpace) {
      for (nht <- nhtSpace) {
        Utils.time {
          println("==========transform training data==========")
          brp = new BucketedRandomProjectionLSH().setBucketLength(bl).setNumHashTables(nht).setInputCol("features").setOutputCol("hashes")
          model = brp.fit(training_df_ml)
          transformedA = model.transform(training_df_ml)
        }
        for (k <- kSpace) {
          Utils.time {
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
