#!/usr/bin/bash


for i in {1..1};
    do
		python -m octopus+ study/e1/e1p1.toml study/e1/e1p1v3.py V --slurm --go
		python -m octopus+ study/e1/e1p2.toml study/e1/e1p2v3.py V --slurm --go
		#python -m octopus+ study/e1/e1p1.toml study/e1/e1p1v3.py V --slurm --go
		#python -m octopus+ study/e1/e1p2.toml study/e1/e1p2v3.py V --slurm --go
		#python -m octopus+ study/e1/e1p3.toml study/e1/e1p3.py V --slurm --go
	done



