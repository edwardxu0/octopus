#!/usr/bin/bash

for i in {2..2};
do
	for j in {1..3};
	do
		echo $i $j;
		echo "python -m octopus+ study/e1/e1p${j}_.toml study/e1/e1p${j}v${i}.py $1 --slurm --go $2"
		python -m octopus+ study/e1/e1p${j}_.toml study/e1/e1p${j}v${i}.py $1 --slurm --go $2
	done
done



