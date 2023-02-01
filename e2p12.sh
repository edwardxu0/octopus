#!/usr/bin/env bash


cmd="python -m octopus+ study_comb/e2/e2p12.toml study_comb/e2/e2p12ov0.py $1 --slurm $2"
echo $cmd
$cmd

