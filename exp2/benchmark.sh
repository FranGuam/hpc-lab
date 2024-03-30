#!/bin/bash
set -e
set -u

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

set -x

for node in 1 2 4; do
    for process in 4 8 16 32 64; do
        for scale in 64 1024 16384 262144 4194304 67108864 1073741824; do
            srun -N $node -n $process ./allreduce 10 $scale
        done
    done
done