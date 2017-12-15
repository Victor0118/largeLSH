package largelsh

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

object DataLoader {

  def getDatasets(dataset: String, sample: Option[Int], sc: SparkContext) = {
    var training = dataset match {
      case "mnist" => MLUtils.loadLibSVMFile(sc, "data/mnist.bz2")
      case "svhn" => MLUtils.loadLibSVMFile(sc, "data/SVHN.bz2")
    }

    var testing = dataset match {
      case "mnist" => MLUtils.loadLibSVMFile(sc, "data/mnist.t.bz2")
      case "svhn" => MLUtils.loadLibSVMFile(sc, "data/SVHN.t.bz2")
    }

    if (sample.isDefined) {
      training = sc.parallelize(training.take(sample.get))
      testing = sc.parallelize(testing.take(sample.get))
    }

    (training, testing)
  }

}
