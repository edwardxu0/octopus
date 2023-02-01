#!/usr/bin/env bash

python -m octopus+ study_10/e1/e1p10.toml study_10/e1/e1p10ov1.py $1 --slurm --go $2
python -m octopus+ study_10/e1/e1p10_.toml study_10/e1/e1p10ov1_.py $1 --slurm --go $2
