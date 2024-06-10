#!/bin/bash

source /home/spack/spack/share/spack/setup-env.sh

spack load gcc@10.2.0
spack load cuda
spack load cmake@3.24.4

mkdir build
cd build
cmake ..
make -j4