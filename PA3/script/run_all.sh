#!/bin/bash

dsets=(arxiv collab citation ddi protein ppa reddit.dgl products youtube amazon_cogdl yelp wikikg2 am)
filename=output_$(date +"%m-%d_%H-%M-%S").log

echo Log saved to $filename

for j in `seq 0 $((${#dsets[@]}-1))`;
do
    echo ${dsets[j]} 32
    srun -N 1 --gres=gpu:1 ./build/test/unit_tests --dataset ${dsets[j]} --len 32 --datadir ./data/ 2>&1 | tee -a ./output/$filename 
    echo ${dsets[j]} 256
    srun -N 1 --gres=gpu:1 ./build/test/unit_tests --dataset ${dsets[j]} --len 256 --datadir ./data/ 2>&1 | tee -a ./output/$filename
done

