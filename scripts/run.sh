#!/usr/bin/env bash

set -x

declare -a classes=("RandomProjection" "RandomProjectionWithDistance")

for class in "${classes[@]}"
do
    for i in $(seq 2 4)
    do
        for j in $(seq 1 4)
        do
            /usr/bin/time spark-submit --num-executors $i --executor-cores $j --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --k 3 --m 10
            /usr/bin/time spark-submit --num-executors $i --executor-cores $j --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --k 5 --m 20
            /usr/bin/time spark-submit --num-executors $i --executor-cores $j --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --k 7 --m 25
            /usr/bin/time spark-submit --num-executors $i --executor-cores $j --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --k 5 --m 10
        done
    done
done

declare -a classes=("SparkLSH")

for class in "${classes[@]}"
do
    /usr/bin/time spark-submit --num-executors 2 --executor-cores 1 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 2 --executor-cores 2 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 2 --executor-cores 3 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 2 --executor-cores 4 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 2 --executor-cores 5 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 3 --executor-cores 1 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 3 --executor-cores 2 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 3 --executor-cores 3 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 3 --executor-cores 4 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 3 --executor-cores 5 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 4 --executor-cores 1 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 4 --executor-cores 2 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 4 --executor-cores 3 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 4 --executor-cores 4 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
    /usr/bin/time spark-submit --num-executors 5 --executor-cores 3 --executor-memory 4g --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j-spark.properties" --class "largelsh.$class" target/scala-2.11/LargeLSH-assembly-0.1.0-SNAPSHOT.jar --dataset svhn --mode eval
done
