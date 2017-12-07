package largelsh

import scala.math._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.PriorityQueue
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{SparseVector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.rogach.scallop._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val seed = opt[Int](default = Some(1234), descr = "Random seed")
  val k = opt[Int](default = Some(20), descr = "Number of hash functions in each set")
  val m = opt[Int](default = Some(10), descr = "Number of sets of hash functions")
  verify()
}

object RandomProjection {
  def getPredictions(buckets: scala.collection.Map[(Seq[Int],Int),ListBuffer[Double]], hashFunctions: Seq[Array[(Array[Double]) => Int]], dataset: RDD[LabeledPoint], k: Int = 5) = {
    val addToSet = (s: ListBuffer[Double], v: Double) => s += v
    val mergePartitionSets = (s1: ListBuffer[Double], s2: ListBuffer[Double]) => s1 ++= s2
    dataset.map(p => {
      val featuresArray = p.features.toArray
      val signatures = hashFunctionSets.map(hashFunctions => {
        hashFunctions.map(f => f(featuresArray)).toSeq
      })

      val labelsInBucket = signatures
        .zipWithIndex
        .map(k => buckets.getOrElse(k, ListBuffer.empty[Double]))
        .reduce((lb1, lb2) => lb1 ++= lb2)

      val labelsInBucketList = labelsInBucket.toList
      val mostCommonLabel = if (!labelsInBucketList.isEmpty) labelsInBucketList.groupBy(identity).mapValues(_.size).maxBy(_._2)._1 else -1
      (p.label, mostCommonLabel)
    })
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
    val sparkConf = new SparkConf().setAppName("LSH using Random Projection").setMaster("local")
    val sc = new SparkContext(sparkConf)

    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    val trainingNumFeats = training.take(1)(0).features.size
    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")

    // Pad testing examples to make even with training examples
    val testingPadded = testing.map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(trainingNumFeats, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    })

    // Generate hash functions for each set
    val hashFunctionSets = (1 to m).map(setNum => {
      val random = new Random(seed + setNum)
      val hashFunctions = Array.fill(k)(Utils.getRandomProjectionHashFunction(random, trainingNumFeats))
      hashFunctions
    })

    // Generate signatures for training set
    val addToSet = (s: ListBuffer[Double], v: Double) => s += v
    val mergePartitionSets = (s1: ListBuffer[Double], s2: ListBuffer[Double]) => s1 ++= s2
    val buckets = training.flatMap(p => {
      val featuresArray = p.features.toArray
      val setSignatures = hashFunctionSets.map(hashFunctions => {
        hashFunctions.map(f => f(featuresArray)).toSeq
      })
      val emitKVPs = setSignatures.zipWithIndex.map(k => (k, p.label))
      // Emit [((signature of bucket, bucket id), label)]
      emitKVPs
    }).aggregateByKey(ListBuffer.empty[Double])(addToSet, mergePartitionSets)
    .collectAsMap

    val trainPredictions = getPredictions(buckets, hashFunctionSets, training)
    val testPredictions = getPredictions(buckets, hashFunctionSets, testing)

    println(s"Training accuracy: ${getAccuracy(trainPredictions)}")
    println(s"Test accuracy: ${getAccuracy(testPredictions)}")
  }
}
