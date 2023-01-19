#!/usr/bin/bash

for i in {2..2};
do
    python -m octopus+ study_conv/e1/e1p4v3.toml study_conv/e1/e1p4v${i}_oe.py $1 --slurm --go
done
