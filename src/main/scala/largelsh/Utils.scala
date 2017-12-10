package largelsh

import scala.math._
import scala.collection.mutable.{HashMap,ListBuffer}
import scala.util.Random

import breeze.linalg.StorageVector
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector,Vector}
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

  def toBreeze(vector: Vector) : breeze.linalg.Vector[scala.Double] = vector match {
    case sv: SparseVector => new breeze.linalg.SparseVector[Double](sv.indices, sv.values, sv.size)
    case dv: DenseVector => new breeze.linalg.DenseVector[Double](dv.values)
  }

  def getRandomProjectionHashFunction(random: Random, dim: Int) : (breeze.linalg.Vector[Double]) => Int = {
    val a = breeze.linalg.DenseVector.fill(dim){random.nextGaussian}

    def hashFunction(v: breeze.linalg.Vector[Double]): Int =  {
      val dotProduct = a dot v
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
