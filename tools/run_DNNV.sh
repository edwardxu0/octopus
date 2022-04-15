#!/bin/bash
cd $DNNV
. .venv/bin/activate
cd - > /dev/null
python -m dnnv $@
