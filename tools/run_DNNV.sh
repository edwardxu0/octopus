#!/bin/bash
cd $DNNV
. .venv/bin/activate
cd -
python -m dnnv $@
