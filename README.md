# largeLSH
Course Project for CS 798

Presentation Link: https://docs.google.com/presentation/d/14DKWcGOVzM7ybIG6Uv3ZF-jqdMN-D-gJOFD9DOFp1sE/edit?usp=sharing

Experiment Data Spreadsheet: https://docs.google.com/spreadsheets/d/1AwBqONysRTfGpHeHmLJdd-U8EXq-Mtz17rj5TwPjhoY/edit?usp=sharing

Code used for experimenting with spill tree implementation (with modifications from author's work): https://github.com/tuzhucheng/spark-knn/blob/largelsh-compare/spark-knn-examples/src/main/scala/com/github/saurfang/spark/ml/knn/examples/CustomBenchmark.scala

How do I use it?

Download data files by running `download.sh` in data/

`sbt assembly` to create a jar.

Send it to Spark!

```
/usr/bin/time spark-submit --num-executors 4 --executor-cores 2 --executor-memory 24g --class largelsh.SparkLSHv2 target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --mode evaluate --k 3 --dataset mnist
```


## Related Work

1. Code
* https://github.com/richwhitjr/DistNN
* https://github.com/zbweng/Distributed-LSH
* https://github.com/mrsqueeze/spark-hash
* https://github.com/marufaytekin/lsh-spark

2. Paper
* LSH At Large: http://people.csail.mit.edu/pcm/papers/LSHatLarge.pdf
* Efficient: https://arxiv.org/abs/1210.7057
* General Metric Data: http://eduardovalle.com/wordpress/wp-content/uploads/2014/10/silva14sisapLargeScaleMetricLSH.pdf
