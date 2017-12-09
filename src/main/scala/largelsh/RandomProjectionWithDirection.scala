package largelsh

import scala.math._
import scala.collection.mutable.{HashMap,ListBuffer,PriorityQueue}
import scala.util.Random

import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{SparseVector,Vector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.rogach.scallop._

object RandomProjectionWithDirection {
  def getPredictions(buckets: scala.collection.Map[(Seq[Int],Int),ListBuffer[(Double,Long)]], indexToFeatureVec: scala.collection.Map[Long,Vector], hashFunctionSets: Seq[Array[(Array[Double]) => Int]], dataset: RDD[LabeledPoint], numNearestNeighbour: Int = 5) = {
    dataset.zipWithIndex.map{ case (p, j) => {
      val featuresArray = p.features.toArray
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
    val sparkConf = new SparkConf().setAppName("LSH using Random Projection with Direction").setMaster("local")
    val sc = new SparkContext(sparkConf)

    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    val trainingNumFeats = training.take(1)(0).features.size
    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")

    // Pad testing examples to make even with training examples
    val testingPadded = Utils.pad(testing, trainingNumFeats)

    // Generate hash functions for each set
    val hashFunctionSets = Utils.generateHashFunctionSets(m, k, trainingNumFeats, seed)

    // Build mapping from index to feature vector
    val indexToFeatureVec = training.zipWithIndex.map {
      case (e, i) => (i, e.features)
    }.collectAsMap

    // Generate signatures for training set
    val addToSet = (s: ListBuffer[(Double,Long)], v: (Double,Long)) => s += v
    val mergePartitionSets = (s1: ListBuffer[(Double,Long)], s2: ListBuffer[(Double,Long)]) => s1 ++= s2
    val buckets = training.zipWithIndex.flatMap {
      case (p, i) => {
        val featuresArray = p.features.toArray
        val setSignatures = hashFunctionSets.map(hashFunctions => {
          hashFunctions.map(f => f(featuresArray)).toSeq
        })
        val emitKVPs = setSignatures.zipWithIndex.map(k => (k, (p.label, i)))
        // Emit [((signature of bucket, bucket id), (label, exampleIdx))]
        emitKVPs
      }
    }.aggregateByKey(ListBuffer.empty[(Double, Long)])(addToSet, mergePartitionSets)
    .collectAsMap

    val trainPredictions = getPredictions(buckets, indexToFeatureVec, hashFunctionSets, training)
    val testPredictions = getPredictions(buckets, indexToFeatureVec, hashFunctionSets, testing)

    val trainAccuracy = getAccuracy(trainPredictions)
    val testAccuracy = getAccuracy(testPredictions)
    sc.stop()

    println(s"Training accuracy: ${trainAccuracy}")
    println(s"Test accuracy: ${testAccuracy}")
  }
}
