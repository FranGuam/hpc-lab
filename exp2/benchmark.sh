#!/bin/bash
set -e
set -u
set -x

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

for node in 1 2 4; do
    for process in 4 8 16 32 64; do
        for scale in 1000000 10000000 100000000 1000000000 10000000000; do
            srun -N $node -n $process ./allreduce 10 $scale
        done
    done
done