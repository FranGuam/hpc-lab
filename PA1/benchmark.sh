#!/bin/bash
set +e

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

make -j4

data_size=$((seq 2 0.1 9.3 | awk '{printf "%d ", 10^$1}'))
for i in $data_size; do
    ./generate $i ./$i.dat random
    ./run.sh ./odd_even_sort $i ./$i.dat
    rm ./$i.dat
    echo " "
done