#!/bin/bash

rm -rf $CONDA_HOME/envs/swarmhost

conda env create -n swarmhost -f lib/SwarmHost/.env.d/env.yml

