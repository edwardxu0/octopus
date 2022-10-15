#!/usr/bin/bash

for i in {1..1};
do
	echo $i;
	sleep 1;
done	

for i in {5..8};
do
	for j in {3..3};
	do
		echo $i $j;
		python -m octopus+ study/e2/e2p${j}_.toml study/e2/e2p${j}v${i}_.py $1 --slurm --go $2
	done
done



