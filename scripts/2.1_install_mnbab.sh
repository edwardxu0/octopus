#!/bin/bash

rm -rf $CONDA_HOME/envs/mnbab

conda env create -n mnbab -f lib/SwarmHost/envs/mnbab.yml
eval "$(conda shell.bash hook)"
conda activate mnbab

octopus_dir=`pwd`
cd $CONDA_HOME/envs/mnbab/include/cddlib
cp * ..
cd $octopus_dir

cd lib/SwarmHost/lib/mnbab/ELINA
make clean
./configure -use-deeppoly -use-fconv --prefix $CONDA_HOME/envs/mnbab -gmp-prefix $CONDA_HOME/envs/mnbab -mpfr-prefix $CONDA_HOME/envs/mnbab -cdd-prefix $CONDA_HOME/envs/mnbab
make
make install
cd ../../../../../