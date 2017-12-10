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

object PairwiseNaive {
  def getPredictions(totalTrainingExamples: Long, indexToLabelAndFeatureVec: scala.collection.Map[Long,(Double,Vector)], dataset: RDD[LabeledPoint], numNearestNeighbour: Int = 5) = {
    val addToSet = (s: ListBuffer[Long], v: Long) => s += v
    val mergePartitionSets = (s1: ListBuffer[Long], s2: ListBuffer[Long]) => s1 ++= s2

    val neighboursList = dataset.zipWithIndex.flatMap {
      case (p, i) => {
        val pairs = for (j <- 0L until totalTrainingExamples if j != i) yield (i, j)
        pairs
      }
    }.aggregateByKey(ListBuffer.empty[Long])(addToSet, mergePartitionSets)

    val pointsToPredict = dataset.zipWithIndex.map { case (lp, i) => (i, lp) }

    val predictions = pointsToPredict.cogroup(neighboursList)
    .map {
      case (k, (lp, v)) => {
        // we expect exactly one LabeledPoint and one ListBuffer
        val lpFeatures = lp.head.features
        var heap = PriorityQueue()(Ordering.by[(Double,Double),Double](_._2).reverse)
        v.head.toList.foreach { j => {
          val (neighbourLabel, neighbourVec) = indexToLabelAndFeatureVec(j)
          val dist = Vectors.sqdist(lpFeatures, neighbourVec)
          heap += ((neighbourLabel, dist))
        }}

        var kClosest = new ListBuffer[Double]()
        for {
          i <- 1 to numNearestNeighbour
          if !heap.isEmpty
        } {
          kClosest += heap.dequeue._1
        }
        val mostCommonLabel = if (!kClosest.isEmpty) kClosest.toList.groupBy(identity).mapValues(_.size).maxBy(_._2)._1 else -1
        (lp.head.label, mostCommonLabel)
      }
    }
    predictions
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
      .appName("Naive All Pairs Implementation")
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

    // Build mapping from index to label and feature vector
    val indexToLabelAndFeatureVec = training.zipWithIndex.map {
      case (e, i) => (i, (e.label, e.features))
    }.collectAsMap

    // Compute distance between all pairs
    val totalTrainingExamples = training.count

    val trainPredictions = getPredictions(totalTrainingExamples, indexToLabelAndFeatureVec, training)
    val testPredictions = getPredictions(totalTrainingExamples, indexToLabelAndFeatureVec, testingPadded)

    val trainAccuracy = getAccuracy(trainPredictions)
    val testAccuracy = getAccuracy(testPredictions)
    sc.stop()

    println(s"Training accuracy: ${trainAccuracy}")
    println(s"Test accuracy: ${testAccuracy}")
  }
}
