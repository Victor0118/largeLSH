package largelsh

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{row_number, _}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.rogach.scallop._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{array, collect_list}
import org.apache.spark.sql.functions.udf

import scala.collection.mutable.WrappedArray

class SparkLSHSiftConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val bl = opt[Double](default = Some(2.0), descr = "Bucket length")
  val nht = opt[Int](default = Some(3), descr = "Number of hash tables")
  val k = opt[Int](default = Some(100), descr = "Number of nearest neighbor in k-NN")
  val sample = opt[Int](default = None, descr = "Run on sample")
  val dataset = opt[String](default = Some("mnist"), descr = "Dataset to run on, mnist or svhn")
  val mode = opt[String](default = Some("eval"), descr = "Use eval to run on parameters provided, search to search over space of parameters")
  verify()
}


object SparkLSHSift {

  def main(args: Array[String]) {
    val conf = new SparkLSHConf(args)
    val spark: SparkSession = SparkSession.builder().appName("LargeLSH").getOrCreate()

    var df = spark.read.option("header", "true").option("inferSchema", "true").csv("data/sift/query.csv")
    var assembler = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
    val query = assembler.transform(df).select("features")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv("data/sift/base.csv")
    val base = assembler.transform(df).select("features")
    df = spark.read.option("inferSchema", "true").csv("data/sift/groundtruth.csv")
    assembler = new VectorAssembler().setInputCols(df.columns).setOutputCol("features")
//    val groundtruth = assembler.transform(df).select("features")
    val groundtruth = df.withColumn("features", array(df.columns.head, df.columns.tail: _*)).select("features")

    //    monotonically_increasing_id is not stable
    //    val df_sample_id = df_sample.withColumn("UniqueID", monotonically_increasing_id)
    //    val base_id = base.withColumn("UniqueID", monotonically_increasing_id)

    val query_id = query.withColumn("id", row_number.over(Window.partitionBy(lit(1)).orderBy(lit(1))))
    val df_sample_id = query_id.sample(false, 1)
    val base_id = base.withColumn("id", row_number.over(Window.partitionBy(lit(1)).orderBy(lit(1))))
    val groundtruth_id = groundtruth.withColumn("id", row_number.over(Window.partitionBy(lit(1)).orderBy(lit(1))))
//    val vectorHead = udf{ x:DenseVector => x(0) }
//    groundtruth_id.withColumn("gt_array", vectorHead(df("features")))

    /**
      * threshold: max l2 distance to filter before sorting
      * bl: BucketLength, W in LSH at large paper
      * nht: number of HashTables
      * k: number of nearest neighbor in k-NN
      */
    val bl = conf.bl()
    val nht = conf.nht()
    val k = conf.k()
    val trainingNumFeats = 128
    val threshold = trainingNumFeats * 2.5
    var brp = new BucketedRandomProjectionLSH().setBucketLength(2).setNumHashTables(2).setInputCol("features").setOutputCol("hashes")
    var model = brp.fit(base)
    var transformedA = model.transform(base_id)

    val searchMode = conf.mode() == "search"
    val blSpace = if (searchMode) Seq(2.0, 5.0, 8.0) else Seq(conf.bl())
    val nhtSpace = if (searchMode) Seq(3, 5, 7) else Seq(conf.nht())
    val kSpace = if (searchMode) Seq(1, 5, 9) else Seq(conf.k())



    for (bl <- blSpace) {
      for (nht <- nhtSpace) {
        Utils.time {
          println("==========transform training data==========")
          brp = new BucketedRandomProjectionLSH().setBucketLength(bl).setNumHashTables(nht).setInputCol("features").setOutputCol("hashes")
          model = brp.fit(base)
          transformedA = model.transform(base)
        }
        for (k <- kSpace) {
          Utils.time {
            println("==========run kNN on testing data==========")
            // Compute the locality sensitive hashes for the input rows, then perform approximate similarity join.
            val results = model.approxSimilarityJoin(transformedA, df_sample_id, threshold, "EuclideanDistance")

            results.createOrReplaceTempView("results")
            val sqlDF = spark.sql("SELECT datasetA.id as trainID, datasetB.id as testID, EuclideanDistance FROM results ")

            // choose the majority from first k candidates
            val w = Window.partitionBy($"testID").orderBy($"EuclideanDistance".asc)
            val dfTopk = sqlDF.withColumn("rn", row_number.over(w)).where($"rn" <= k).drop("rn")

            val flatten = udf((xs: Seq[Seq[Double]]) => xs.flatten)
            val trackingIds = flatten(collect_list($"trainID"))
            val prediction = dfTopk.groupBy($"testID").agg(collect_list("trainID"))
            prediction.join(groundtruth_id).where($"testID" === $"id")

            val same_elements = udf { (a: WrappedArray[String],
                                       b: WrappedArray[String]) =>
              if (a.intersect(b).length == b.length){ 1 }else{ 0 }
            }
            df.withColumn("test",same_elements(col("genreList1"),col("genreList2")))
//            intersect(select(newHiresDF, 'name'), select(salesTeamDF,'name'))
            val interscetion = prediction.sort("testID").select($"collect_list(trainID)").intersect(groundtruth.select($"features"))
//            val finalDF = dfTopk.map(t => {
//              val acc = if (t.getAs(0) == t.getAs(1)) 1 else 0
//              (acc: Int, t.getAs(1): Double, t.getAs(2): org.apache.spark.ml.linalg.SparseVector, t.getAs(3): Double)
//            })
//
//            // group by testing samples and compute the accuracy
//            val scores = finalDF.groupBy("_3").agg(sum("_1") / count("_1"))
//            val accuracy = scores.map(t => if (t.getDouble(1) > 0.5) 1 else 0).reduce { (x, y) => x + y }.toDouble / df_sample.count()
//            println("bl:", bl, "nht:", nht, "k:", k, "accuracy:", accuracy)
          }

        }
      }
    }
  }
}
