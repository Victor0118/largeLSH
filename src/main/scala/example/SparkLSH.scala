
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.runtime.universe._

object Example {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("LSH at Large").setMaster("local")
    val sc = new SparkContext(sparkConf)
    sparkConf.set("spark.driver.allowMultipleContexts", "true")

    val spark: SparkSession = SparkSession.builder().appName("LSH at Large").config("spark.master", "local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // load mnist dataset using mllib library
//    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/w85yang/largeLSH/data/mnist.bz2")
    val trainingNumFeats = training.take(1)(0).features.size

    // change RDD type with mllib Vector to DataFrame type with ml Vector
    val training_df = training.toDF()
    val training_df_ml = MLUtils.convertVectorColumnsToML(training_df)

//    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")
    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/w85yang/largeLSH/data/mnist.t.bz2")
    val testingNumFeats = testing.take(1)(0).features.size

//    val t = testing.map(x => (x.features.toArray ++ Array(0.0, 0.0)).toVector)
//    val t2 = testing.map(x => {new SparseVector(trainingNumFeats, x.features.toSparse.indices, x.features.toSparse.values))
    val testing_padded = testing.map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(trainingNumFeats, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    })

    val testing_df = testing_padded.toDF()
    val testing_df_ml = MLUtils.convertVectorColumnsToML(testing_df)
    training_df_ml.select("features").show()
    testing_df_ml.select("features").show()


    training_df_ml.select($"features").show()

    val df_sample = testing_df_ml.sample(false, 0.0001)

    val brp = new BucketedRandomProjectionLSH().setBucketLength(2.0).setNumHashTables(3).setInputCol("features").setOutputCol("hashes")
    val model = brp.fit(training_df_ml)


    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    val transformedA = model.transform(training_df_ml)

    println("train to train LSH:")
    model.approxSimilarityJoin(training_df_ml, training_df_ml.limit(10), 1.5, "EuclideanDistance").show()


    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // but the dimension does not match, TO BE FIXED

    println("test to train LSH:")
    val results = model.approxSimilarityJoin(transformedA, df_sample, 2200, "EuclideanDistance")
    results.printSchema()

    results.createOrReplaceTempView("results")
    val sqlDF = spark.sql("SELECT datasetA.label FROM results ")
    sqlDF.show()

    results.sort("EuclideanDistance").groupBy("datasetB").agg(count("datasetA").alias("itemCount"))
//    results.map{case Row(da: struct<label: double, features: vector> features: vector, db:LabeledPoint, dc:Float) => {da.label}}
    results.map{t => {t._1.label}}
  }
}