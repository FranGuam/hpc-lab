#!/bin/bash
set -e
set -u

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

set -x

for node in 1 2 4; do
    for process in 7 14 28 56 112; do
        if [ $(( $process / $node )) -gt 28 ]; then
            echo "Skip $node nodes with $process processes"
            continue
        fi
        for scale in 112 1120 11200 112000 1120000 11200000 112000000; do
            srun -N $node -n $process ./allreduce 10 $scale
        done
    done
done