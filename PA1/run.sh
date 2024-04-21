#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
if [ $# -neq 3 ]; then
    echo "Usage: ./run.sh <executable> <data_size> <data_file>"
    exit 1
fi

executable=$1
data_size=$2
data_file=$3

if [ ! -f $executable ]; then
    echo "Executable $executable not found!"
    exit 1
fi
if [ ! -f $data_file ]; then
    echo "Data file $data_file not found!"
    exit 1
fi
if [ data_size -lt 0 ]; then
    echo "Data size must be positive!"
    exit 1
fi
if [ data_size -gt 2147483647 ]; then
    echo "Data size must be less than 100000000!"
    exit 1
fi

if [ data_size -lt 10000 ]; then
    process=1
else
    process=56
    while [ process -gt 6 ]; do
        block_size=$(((data_size + process - 1) / process))
        last_block_len=$((data_size % block_size))
        if [ block_size % 2 -eq 0 -a last_block_len * 2 -ge block_size]; then
            break
        fi
        process=$((process - 1))
    done
    if [ process -eq 6 ]; then
        process=56
        while [ process -gt 6 ]; do
            block_size=$(((data_size + process - 1) / process))
            if [ block_size % 2 -eq 0 ]; then
                break
            fi
            process=$((process - 1))
        done
    fi
    if [ process -eq 6 ]; then
        process=28
    fi
fi
node=1
if [ process -gt 28 ]; then
    node=2
fi

echo "Running $data_size data with $process process"
srun -N $node -n $process $executable $data_size $data_file
