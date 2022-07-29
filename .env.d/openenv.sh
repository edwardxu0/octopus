#!/bin/bash

# activate virtual environment
source .venv/bin/activate

# add library to paths
export DNNV=./lib/DNNV
export DNNVWB=./lib/DNNVWB
export SIP=./lib/SIP

export PYTHONPATH=$PYTHONPATH:$SIP

# Remove tensorflow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# aliasing tool
alias octopus='python -m octopus'
alias octopus+='python -m octopus+'
export PYTHONHASHSEED=0

