#!/bin/bash

source /home/spack/spack/share/spack/setup-env.sh

spack load cuda && spack load gcc@10.2.0

rm -f ./benchmark

make

set -x

srun -N 1 --gres=gpu:1 ./benchmark 100
srun -N 1 --gres=gpu:1 ./benchmark 1000
srun -N 1 --gres=gpu:1 ./benchmark 2500
srun -N 1 --gres=gpu:1 ./benchmark 5000
srun -N 1 --gres=gpu:1 ./benchmark 7500
srun -N 1 --gres=gpu:1 ./benchmark 10000

srun -N 1 --gres=gpu:1 nvprof --profile-from-start off ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --dependency-analysis ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_ld_bank_conflict ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_st_bank_conflict ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics achieved_occupancy ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics gld_throughput ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics gld_efficiency ./benchmark 100
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics ipc ./benchmark 100

srun -N 1 --gres=gpu:1 nvprof --profile-from-start off ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --dependency-analysis ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_ld_bank_conflict ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_st_bank_conflict ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics achieved_occupancy ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics gld_throughput ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics gld_efficiency ./benchmark 1000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics ipc ./benchmark 1000

srun -N 1 --gres=gpu:1 nvprof --profile-from-start off ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --dependency-analysis ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_ld_bank_conflict ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --events shared_st_bank_conflict ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics achieved_occupancy ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics gld_throughput ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics gld_efficiency ./benchmark 10000
srun -N 1 --gres=gpu:1 nvprof --profile-from-start off --metrics ipc ./benchmark 10000