#!/usr/bin/bash

python -m octopus+ study/e5/e5p3a.toml study/e5/e5p3a.py $1 --slurm --go
python -m octopus+ study/e5/e5p3b.toml study/e5/e5p3b.py $1 --slurm --go

python -m octopus+ study/e6/e6p3a.toml study/e6/e6p3a.py $1 --slurm --go
python -m octopus+ study/e6/e6p3b.toml study/e6/e6p3b.py $1 --slurm --go

python -m octopus+ study/e7/e7p3a.toml study/e7/e7p3a.py $1 --slurm --go
python -m octopus+ study/e7/e7p3b.toml study/e7/e7p3b.py $1 --slurm --go

python -m octopus+ study/e8/e8p3a.toml study/e8/e8p3a.py $1 --slurm --go
python -m octopus+ study/e8/e8p3b.toml study/e8/e8p3b.py $1 --slurm --go