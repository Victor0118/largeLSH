package largelsh

import scala.math._
import scala.util.Random

import org.apache.spark.mllib.linalg.{SparseVector,Vectors}

object Utils {
  def getRandomProjectionHashFunction(random: Random, dim: Int) : (Array[Double]) => Int = {
    val a = Array.fill(dim)(random.nextGaussian)
    val length = sqrt(a.map(e => e*e).sum)
    val normalized = a.map(e => e / length)

    def hashFunction(v: Array[Double]): Int =  {
      val dotProduct = (a zip v).map(p => p._1 * p._2).sum
      if (dotProduct >= 0) 1 else 0
    }

    hashFunction
  }

  def l2DistanceSquared(p1: SparseVector, p2: SparseVector) : Double = {
    val sharedIndices = p1.indices intersect p2.indices
    sharedIndices.map(i => pow(p1(i) - p2(i), 2)).sum
  }
}
