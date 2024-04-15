#!/bin/bash
set -e
set -u

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

make -j4

echo "Test on Provided Data"
for i in 100 1000 10000 100000 1000000 10000000 100000000; do
    for j in {1..28}; do
        echo "Testing $i data with 1 node and $j process"
        srun -N 1 -n $j ./odd_even_sort $i data/$i.dat
        echo " "
    done
    for j in {29..56}; do
        echo "Testing $i data with 2 node and $j process"
        srun -N 2 -n $j ./odd_even_sort $i data/$i.dat
        echo " "
    done
done

echo "Test on Descending Data"
for i in 31 1000 23459 8972407 112000000; do
    ./generate $i ./$i.dat
    for j in {1..28}; do
        echo "Testing $i data with 1 node and $j process"
        srun -N 1 -n $j ./odd_even_sort $i ./$i.dat
        echo " "
    done
    for j in {29..56}; do
        echo "Testing $i data with 2 node and $j process"
        srun -N 2 -n $j ./odd_even_sort $i data/$i.desc
        echo " "
    done
    rm ./$i.dat
done