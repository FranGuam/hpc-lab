#!/bin/bash
set -e
set -u

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

set -x

for node in 1 2 4; do
    for process in 4 8 16 32 64; do
        for scale in 256 2048 16384 131072 1048576 8388608 67108864 536870912; do
            srun -N $node -n $process ./allreduce 10 $scale
        done
    done
done