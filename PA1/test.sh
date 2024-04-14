#!/bin/bash
set -e
set -u

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

set -x

echo "Testing Provided Data"
for k in 100 1000 10000 100000 1000000 10000000 100000000; do
    for i in 1 2; do
        for j in {1..28}; do
            srun -N $i -n $j ./odd_even_sort $k data/$k.dat
        done
    done
done