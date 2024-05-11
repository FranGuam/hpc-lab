#!/bin/bash

source /opt/intel/oneapi/setvars.sh

rm -f ./main
make UNROLL_N=1
srun -n 4 ./main

rm -f ./main
make UNROLL_N=2
srun -n 4 ./main

rm -f ./main
make UNROLL_N=4
srun -n 4 ./main

rm -f ./main
make UNROLL_N=8
srun -n 4 ./main

rm -f ./main
make UNROLL_N=16
srun -n 4 ./main
