package largelsh

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

object DataLoader {

  def getDatasets(dataset: String, sample: Option[Int], sc: SparkContext) = {
    var training = dataset match {
      case "mnist" => MLUtils.loadLibSVMFile(sc, "data/mnist")
      case "svhn" => MLUtils.loadLibSVMFile(sc, "data/SVHN")
    }

    var testing = dataset match {
      case "mnist" => MLUtils.loadLibSVMFile(sc, "data/mnist.t")
      case "svhn" => MLUtils.loadLibSVMFile(sc, "data/SVHN.t")
    }

    if (sample.isDefined) {
      training = sc.parallelize(training.take(sample.get))
      testing = sc.parallelize(testing.take(sample.get))
    }

    (training, testing)
  }

}
