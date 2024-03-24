#!/bin/bash

set -x

srun -n 2 -N 2 --export=ALL,UCX_NET_DEVICES=eno1 /home/course/hpc/tools/osu_latency -m :1048576