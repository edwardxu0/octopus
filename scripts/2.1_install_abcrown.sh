#!/bin/bash

rm -rf $CONDA_HOME/envs/abcrown

conda env create -n abcrown -f lib/SwarmHost/envs/abcrown.yml