#!/bin/bash

# activate virtual environment
#source .venv/bin/activate
conda activate octopus

# add library to paths
export OCTOPUS=`pwd`
export ROOT=`pwd`
export SIP=$OCTOPUS/lib/SIP
export ALR=$OCTOPUS/lib/auto_LiRPA
export SwarmHost=$OCTOPUS/lib/SwarmHost
export NeuralSAT=$OCTOPUS/lib/neuralsat

export PYTHONPATH=$PYTHONPATH:$SIP:$ALR:$SwarmHost
export PYTHONPATH=$PYTHONPATH:$SIP:$ALR:$SwarmHost/lib/verinet
export PYTHONPATH=$PYTHONPATH:$SIP:$ALR:$NeuralSAT/neuralsat

# Remove tensorflow warnings
export TF_CPP_MIN_LOG_LEVEL=2
export MKL_SERVICE_FORCE_INTEL=1

# aliasing tool
alias octopus='python -m octopus'
alias octopus+='python -m octopus+'
export PYTHONHASHSEED=0

