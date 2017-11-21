#!/usr/bin/env bash

wget https://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/multiclass/mnist.bz2
wget https://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/multiclass/mnist.t.bz2

bzip2 -d -v mnist.bz2
bzip2 -d -v mnist.t.bz2
