package largelsh

import scala.math._
import scala.collection.mutable.{HashMap,ListBuffer}
import scala.util.Random

import org.apache.spark.mllib.linalg.{SparseVector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object Utils {

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1000000.0 + "ms")
    result
  }
  
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

  def pad(dataset: RDD[LabeledPoint], dim: Int) = {
    val result = dataset.map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(dim, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    })
    result
  }

  def generateHashFunctionSets(m: Int, k: Int, hashFunctionDim: Int, seed: Int) = {
    // Generate hash functions for each set
    val hashFunctionSets = (1 to m).map(setNum => {
      val random = new Random(seed + setNum)
      val hashFunctions = Array.fill(k)(getRandomProjectionHashFunction(random, hashFunctionDim))
      hashFunctions
    })
    hashFunctionSets
  }

  def l2DistanceSquared(p1: SparseVector, p2: SparseVector) : Double = {
    val sharedIndices = p1.indices intersect p2.indices
    sharedIndices.map(i => pow(p1(i) - p2(i), 2)).sum
  }

  def addToHashMapCounter(m: HashMap[Double,Int], v: Double) = {
    if (m contains v) {
      m(v) += 1
    } else {
      m += (v -> 0)
    }
    m
  }

  def mergeHashMapCounters(m1: HashMap[Double,Int], m2: HashMap[Double,Int]) = {
    m2.foreach {
      case (k, v) => {
        if (m1 contains k) {
          m1(k) += v
        } else {
          m1 += (k -> v)
        }
      }
    }
    m1
  }
}
