#!/bin/bash

dataset=$1
len=$2

srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --print-gpu-summary ./test/unit_tests --dataset $1 --len $2 --datadir ./data/
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --print-gpu-trace ./test/unit_tests --dataset $1 --len $2 --datadir ./data/
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_ld_bank_conflict,shared_st_bank_conflict ./test/unit_tests --dataset $1 --len $2 --datadir ./data/
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics achieved_occupancy,gld_throughput,gld_efficiency,ipc ./test/unit_tests --dataset $1 --len $2 --datadir ./data/