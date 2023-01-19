#!/usr/bin/env bash

for i in {2,4,6}
do
echo $i
    cmd="python -m octopus+ study_10/e2/e2p11.toml study_10/e2/e2p11_${i}v1.py $1 --slurm --go $2"
    echo $cmd
    $cmd
done
