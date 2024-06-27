#!/bin/bash

source /home/spack/spack/share/spack/setup-env.sh

spack load cuda

dataset=$1
len=$2

srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --print-gpu-summary ./build/test/unit_tests --dataset $1 --len $2 --datadir ./data/
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --print-gpu-trace ./build/test/unit_tests --dataset $1 --len $2 --datadir ./data/
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_ld_bank_conflict,shared_st_bank_conflict ./build/test/unit_tests --dataset $1 --len $2 --datadir ./data/
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics achieved_occupancy,gld_throughput,gld_efficiency,ipc ./build/test/unit_tests --dataset $1 --len $2 --datadir ./data/