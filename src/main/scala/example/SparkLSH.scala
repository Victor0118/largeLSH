
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Example {
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
    val testing_df = testing.toDF()
    val testing_df_ml = MLUtils.convertVectorColumnsToML(testing_df)
    training_df_ml.select("features").show()
    testing_df_ml.select("features").show()

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = brp.fit(training_df_ml)

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(training_df_ml).show()

    println("train to train LSH:")
    model.approxSimilarityJoin(training_df_ml, training_df_ml.limit(10), 1.5, "EuclideanDistance").show()

    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // but the dimension does not match, TO BE FIXED
//    model.approxSimilarityJoin(training_df_ml, testing_df_ml, 1.5, "EuclideanDistance").show()

  }
}