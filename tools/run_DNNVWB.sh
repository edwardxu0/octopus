#!/bin/bash
cd $DNNVWB
. .venv/bin/activate
cd - > /dev/null
python -m dnnv $@
