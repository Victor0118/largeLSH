package largelsh

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{row_number, _}
import org.rogach.scallop._

object SparkLSHv2 {

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

    val testing_padded = testing.map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(trainingNumFeats, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    })

    val testing_df = testing_padded.toDF()
    val testing_df_ml = MLUtils.convertVectorColumnsToML(testing_df)
    training_df_ml.select("features").show()
    testing_df_ml.select("features").show()

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
    var transformedB = model.transform(testing_df_ml)

    val searchMode = conf.mode() == "search"
    val blSpace = if (searchMode) Seq(2.0, 5.0, 8.0) else Seq(conf.bl())
    val nhtSpace = if (searchMode) Seq(3, 5, 7) else Seq(conf.nht())
    val kSpace = if (searchMode) Seq(1, 5, 9) else Seq(conf.k())

    for (bl <- blSpace) {
      for (nht <- nhtSpace) {
        var testingCount = 0L
        Utils.time {
          println("==========transform training data==========")
          brp = new BucketedRandomProjectionLSH()
                     .setBucketLength(bl)
                     .setNumHashTables(nht)
                     .setInputCol("features")
                     .setOutputCol("hashes")

          model = brp.fit(training_df_ml)
          transformedA = model.transform(training_df_ml).cache
          transformedB = model.transform(testing_df_ml).cache
          testingCount = transformedB.count
        }
        for (k <- kSpace) {
          Utils.time {
            println("==========run kNN on testing data==========")
            // Compute the locality sensitive hashes for the input rows, then perform approximate similarity join.
            model.approxSimilarityJoin(transformedA, transformedB, threshold, "EuclideanDistance")

            val predictionPoints = transformedB.select("label", "features")
                                               .rdd
                                               .zipWithIndex

            val seqop = (s: (Double, Double), t: (Double, Double)) => if (t._1 == t._2) (s._1 + 1, s._2 + 1) else (s._1, s._2 + 1)
            val combop = (s1: (Double, Double), s2: (Double, Double)) => (s1._1 + s2._1, s1._2 + s2._2)
            val groups = testingCount / 1000
            val overallAccAndCount = (0L until groups).toList.par.map(mod => {
              val predictionsSubset = predictionPoints.filter { case (row, idx) => idx % groups == mod }.collect.par
              val accAndCount = predictionsSubset.map { case (row, idx) => {
                val key = row.getAs[org.apache.spark.ml.linalg.SparseVector](1)
                val ann = model.approxNearestNeighbors(transformedA, key, k)
                val prediction = ann.select("label").groupBy("label").count.sort(desc("label")).first.getDouble(0)
                (row.getDouble(0), prediction)  // label, prediction
              }}.aggregate((0.0, 0.0))(seqop, combop)

              accAndCount
            }).aggregate((0.0, 0.0))(combop, combop)
            val accuracy = overallAccAndCount._1 / overallAccAndCount._2
            println("bl:", bl, "nht:", nht, "k:", k, "accuracy:", accuracy)
          }

        }
      }
    }
  }
}
