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
  val seed = opt[Int](default = Some(1234))
  val k = opt[Int](default = Some(20))
  val W = opt[Int](default = Some(10))
  verify()
}

object LSHAtLarge {
  def getHashFunction(random: Random, dim: Int, W: Int) : (Array[Double]) => Double = {
    // N(0, 1) is 2-stable
    val a = Array.fill(dim)(random.nextGaussian)
    val B = random.nextDouble * W

    def hashFunction(v: Array[Double]): Double =  {
      val dotProduct = (a zip v).map(p => p._1 * p._2).sum
      floor((dotProduct + B) / W)
    }

    hashFunction
  }

  def l2DistanceSquared(p1: SparseVector, p2: SparseVector) : Double = {
    val sharedIndices = p1.indices intersect p2.indices
    sharedIndices.map(i => pow(p1(i) - p2(i), 2)).sum
  }

  def getPredictions(buckets: scala.collection.Map[Double,List[(Double,org.apache.spark.mllib.linalg.Vector)]], hashFunctions: Array[(Array[Double]) => Double], dataset: RDD[LabeledPoint], k: Int = 5) = {
    dataset.map(p => {
      val hashes = hashFunctions.map(f => f(p.features.toArray))
      val ksi = hashes.sum
      val heap = PriorityQueue()(Ordering.by[(Double,Double),Double](_._2).reverse)
      var (leftProbe, rightProbe) = (ksi, ksi)
      while (heap.size < k) {
        if (buckets contains leftProbe) {
          heap ++= buckets(leftProbe).map(t => (t._1, l2DistanceSquared(p.features.toSparse, t._2.toSparse)))
        }

        if (buckets contains rightProbe) {
          heap ++= buckets(rightProbe).map(t => (t._1, l2DistanceSquared(p.features.toSparse, t._2.toSparse)))
        }

        leftProbe -= 1
        rightProbe += 1
      }

      var kClosest = new ListBuffer[Double]()
      for (i <- 1 to k) {
        kClosest += heap.dequeue._1
      }
      val mostCommonLabel = kClosest.toList.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
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
    val W = conf.W()
    println(s"Using seed: $seed, k: $k, W: $W")
    val sparkConf = new SparkConf().setAppName("LSG at Large").setMaster("local")
    val sc = new SparkContext(sparkConf)

    val training: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist")
    val trainingNumFeats = training.take(1)(0).features.size
    val testing: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mnist.t")
    val testingPadded = testing.map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(trainingNumFeats, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    })

    val random = new Random(seed)
    val hashFunctions = Array.fill(k)(LSHAtLarge.getHashFunction(random, trainingNumFeats, W))

    val buckets = training.map(p => {
      val hashes = hashFunctions.map(f => f(p.features.toArray))
      val ksi = hashes.sum
      (ksi, List((p.label, p.features)))
    }).reduceByKey(_ ++ _)
    .collectAsMap

    val trainPredictions = getPredictions(buckets, hashFunctions, training)
    val testPredictions = getPredictions(buckets, hashFunctions, testing)

    println(s"Training accuracy: ${getAccuracy(trainPredictions)}")
    println(s"Test accuracy: ${getAccuracy(testPredictions)}")
  }
}
