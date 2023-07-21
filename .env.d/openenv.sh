#!/bin/bash

# activate virtual environment
source .venv/bin/activate

# add library to paths
export DNNV=./lib/DNNV
export DNNVWB=./lib/DNNVWB
export SIP=./lib/SIP
export ALR=./lib/auto_LiRPA
export SwarmHost=./lib/SwarmHost

export PYTHONPATH=$PYTHONPATH:$SIP:$ALR:$SwarmHost

# Remove tensorflow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# aliasing tool
alias octopus='python -m octopus'
alias octopus+='python -m octopus+'
export PYTHONHASHSEED=0

