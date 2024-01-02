#!/bin/bash

rm -rf $CONDA_HOME/envs/nnenum

conda env create -n nnenum -f lib/SwarmHost/envs/nnenum.yml