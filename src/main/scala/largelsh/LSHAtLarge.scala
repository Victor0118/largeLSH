package largelsh

import scala.math._
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
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

        val random = new Random(seed)
        val hashFunctions = Array.fill(k)(LSHAtLarge.getHashFunction(random, trainingNumFeats, W))
    }
}
