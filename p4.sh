#!/usr/bin/bash

for i in {1..3};
do
    python -m octopus+ study_conv/e1/e1p4.toml study_conv/e1/e1p4v$i.py $1 --slurm --go
done
