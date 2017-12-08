package largelsh

import scala.math._
import scala.collection.mutable.{HashMap,ListBuffer}
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{SparseVector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.rogach.scallop._

object RandomProjectionWithDirection {
  def getPredictions(buckets: scala.collection.Map[(Seq[Int],Int),HashMap[Double,Int]], hashFunctionSets: Seq[Array[(Array[Double]) => Int]], dataset: RDD[LabeledPoint], k: Int = 5) = {
    dataset.map(p => {
      val featuresArray = p.features.toArray
      val signatures = hashFunctionSets.map(hashFunctions => {
        hashFunctions.map(f => f(featuresArray)).toSeq
      })

      val labelsInBucket = signatures
        .zipWithIndex
        .map(k => buckets.getOrElse(k, HashMap.empty[Double,Int]))
        .reduce(Utils.mergeHashMapCounters)

      val mostCommonLabel = if (!labelsInBucket.isEmpty) labelsInBucket.maxBy(_._2)._1 else -1
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
    val sparkConf = new SparkConf().setAppName("LSH using Random Projection with Direction").setMaster("local")
    val sc = new SparkContext(sparkConf)

    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    val trainingNumFeats = training.take(1)(0).features.size
    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")

    // Pad testing examples to make even with training examples
    val testingPadded = Utils.pad(testing, trainingNumFeats)

    // Generate hash functions for each set
    val hashFunctionSets = Utils.generateHashFunctionSets(m, k, trainingNumFeats, seed)

    // Generate signatures for training set
    val buckets = training.flatMap(p => {
      val featuresArray = p.features.toArray
      val setSignatures = hashFunctionSets.map(hashFunctions => {
        hashFunctions.map(f => f(featuresArray)).toSeq
      })
      val emitKVPs = setSignatures.zipWithIndex.map(k => (k, p.label))
      // Emit [((signature of bucket, bucket id), label)]
      emitKVPs
    }).aggregateByKey(HashMap.empty[Double,Int])(Utils.addToHashMapCounter, Utils.mergeHashMapCounters)
    .collectAsMap

    val trainPredictions = getPredictions(buckets, hashFunctionSets, training)
    val testPredictions = getPredictions(buckets, hashFunctionSets, testing)

    val trainAccuracy = getAccuracy(trainPredictions)
    val testAccuracy = getAccuracy(testPredictions)
    sc.stop()

    println(s"Training accuracy: ${trainAccuracy}")
    println(s"Test accuracy: ${testAccuracy}")
  }
}
