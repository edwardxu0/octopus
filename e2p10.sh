#!/usr/bin/env bash

for i in {0..0}
do
    #for j in {'o','d','w'}
	for j in {'o'}
    do
		j=o
        cmd="python -m octopus+ study_10/e2/e2p10.toml study_10/e2/e2p10${j}v${i}.py $1 --slurm --go $2"
        echo $cmd
        $cmd
    done
done
