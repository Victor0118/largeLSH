package largelsh

import scala.math._
import scala.collection.mutable.{HashMap,ListBuffer,PriorityQueue}
import scala.util.Random

import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.{SparseVector,Vector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.rogach.scallop._

object RandomProjectionNaive {
  def getPredictions(buckets: scala.collection.Map[(Seq[Int],Int),ListBuffer[(Double,Long)]], indexToFeatureVec: scala.collection.Map[Long,Vector], hashFunctionSets: Seq[Array[(breeze.linalg.Vector[Double]) => Int]], dataset: RDD[LabeledPoint], numNearestNeighbour: Int = 5) = {
    dataset.zipWithIndex.map{ case (p, j) => {
      val featuresArray = Utils.toBreeze(p.features)
      val signatures = hashFunctionSets.map(hashFunctions => {
        hashFunctions.map(f => f(featuresArray)).toSeq
      })

      val labelsInBucket = signatures
        .zipWithIndex
        .map(key => buckets.get(key).getOrElse(ListBuffer.empty[(Double,Long)]))
        .reduce((lb1, lb2) => lb1 ++= lb2)
        .toList

      var heap = PriorityQueue()(Ordering.by[(Double,Double),Double](_._2).reverse)
      labelsInBucket.foreach {
        case (label, i) => {
          val distTup = (label, Vectors.sqdist(indexToFeatureVec.get(i).get, p.features))
          heap += distTup
        }
      }

      var kClosest = new ListBuffer[Double]()
      for {
        i <- 1 to numNearestNeighbour
        if !heap.isEmpty
      } {
        kClosest += heap.dequeue._1
      }
      val mostCommonLabel = if (!labelsInBucket.isEmpty) kClosest.toList.groupBy(identity).mapValues(_.size).maxBy(_._2)._1 else -1
      (p.label, mostCommonLabel)
    }}
  }

  def getAccuracy(predictions: RDD[(Double, Double)]) = {
    val correctPredictions = predictions.map {
      case (label, prediction) => if (label == prediction) 1 else 0
    }.sum

    correctPredictions / predictions.count
  }

  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)
    val seed  = conf.seed()
    val k = conf.k()
    val m = conf.m()
    println(s"Using seed: $seed, k: $k, m: $m")
    val spark = SparkSession
      .builder()
      .appName("LSH using Random Projection with Direction")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    var training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    var testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")
    if (conf.sample.isDefined) {
      training = sc.parallelize(training.take(conf.sample.get.get))
      testing = sc.parallelize(testing.take(conf.sample.get.get))
    }
    val trainingNumFeats = training.take(1)(0).features.size
    println("Number of training features", trainingNumFeats)

    // Pad testing examples to make even with training examples
    val testingPadded = Utils.pad(testing, trainingNumFeats)

    // Build mapping from index to feature vector
    val indexToFeatureVec = training.zipWithIndex.map {
      case (e, i) => (i, e.features)
    }.collectAsMap

    // Compute distance between all pairs
    val addToSet = (s: ListBuffer[Long], v: Long) => s += v
    val mergePartitionSets = (s1: ListBuffer[Long], s2: ListBuffer[Long]) => s1 ++= s2
    val totalTrainingExamples = training.count
    val buckets = training.zipWithIndex.flatMap {
      case (p, i) => {
        val pairs = for (j <- 0 until totalTrainingExamples if j != i) yield (i, j)
      }
    }.aggregateByKey(ListBuffer.empty[Long])(addToSet, mergePartitionSets)
    .map

    val trainPredictions = getPredictions(buckets, indexToFeatureVec, hashFunctionSets, training)
    val testPredictions = getPredictions(buckets, indexToFeatureVec, hashFunctionSets, testingPadded)

    val trainAccuracy = getAccuracy(trainPredictions)
    val testAccuracy = getAccuracy(testPredictions)
    sc.stop()

    println(s"Training accuracy: ${trainAccuracy}")
    println(s"Test accuracy: ${testAccuracy}")
  }
}
