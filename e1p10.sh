#!/usr/bin/env bash



for i in {1..1}
do
	for j in {'o''d','w'}
    do
        cmd="python -m octopus+ study_10/e1/e1p10.toml study_10/e1/e1p10${j}v${i}.py $1 --slurm --go $2"
        echo $cmd
        $cmd
    done
done
