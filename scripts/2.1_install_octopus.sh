#!/bin/bash

rm -rf $CONDA_HOME/envs/octopus

conda env create -n octopus -f .env.d/octopus.yml