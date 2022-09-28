#!/usr/bin/bash

for i in {1..1};
		 do
		 for j in {1..3};
		 do
			 echo $i $j;
			 python -m octopus+ study/e2/e2p${j}.toml study/e2/e2p${j}v${i}.py $1 --slurm --go $2
	     done
	 done



