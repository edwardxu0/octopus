#!/bin/bash

source .venv/bin/activate

export GDVB=./lib/GDVB
export DNNV=./lib/DNNV

export PYTHONPATH=$PYTHONPATH:$GDVB

alias octopus='python -m octopus'
alias octopus+='python -m octopus+'