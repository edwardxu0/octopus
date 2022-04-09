import os

from src.configs import _model_name, _veri_name

from run import MODEL_DIR, PROP_DIR, TRAIN_LOG_DIR, VERI_LOG_DIR

PROP_IMG = {0:28932, 1:42972, 2:49152, 3:49847, 4:2611}

def gen_prop(p,e):
    lines = ['from dnnv.properties import *',
             'import numpy as np',
             '',
             'N = Network("N")',
             f'x = Image("./properties/{PROP_IMG[p]}.npy")',
             'input_layer = 0',
             f'epsilon = {e}',
             'Forall(',
             '    x_,',
             '    Implies(',
             '    ((x - epsilon) < x_ < (x + epsilon)),',
             '        argmax(N[input_layer:](x_)) == argmax(N[input_layer:](x)),',
             '    ),',
             ')']
    lines = [x+'\n' for x in lines]
    open(f'{PROP_DIR}/rb_{p}_{e}.py', 'w').writelines(lines)


def verify_all(strat, problems, nets, itst, ocrc, upbd, props, eps, seeds, verifiers, slurm=False, wb=False, go=False):
    nodes = ['cortado'+x for x in ['01', '02','03','04','05','06','07','08','09','10']]

    model_files = [f'{MODEL_DIR}/{x}' for x  in os.listdir(MODEL_DIR) if 'onnx' in x]
    c = -1
    cc = 0
    for p in problems:
        for n in nets:
            for i in itst:
                for j in ocrc:
                    for k in upbd:
                        for pp in props:
                            for e in eps:
                                for s in seeds:
                                    for v in verifiers:
                                        c += 1
                                        model_path_prefix = _model_name(strat, p, n, s, rurh_itst=i, rurh_ocrc=j, rurh_upbd=k)
                                        model_path = [x for x in model_files if model_path_prefix in x][0]
                                        gen_prop(pp,e)

                                        prop_path = f'{PROP_DIR}/rb_{pp}_{e}.py'
                                        if wb:
                                            cmd = f'python -W ignore ./tools/resmonitor.py -T 600 -M 16G ./tools/run_DNNV_wb.sh {prop_path} --network N {model_path} --{v} --debug'
                                        else:
                                            cmd = f'python -W ignore ./tools/resmonitor.py -T 600 -M 16G ./tools/run_DNNV.sh {prop_path} --network N {model_path} --{v}'
                                        
                                        veri_name = _veri_name(strat, p, n, s, pp, e, v, rurh_itst=i, rurh_ocrc=j, rurh_upbd=k)
                                        out_path = f'{VERI_LOG_DIR}/{veri_name}.txt'

                                        if os.path.exists(out_path):
                                            continue
                                        if slurm:
                                            lines = ['#!/bin/sh',
                                            f'#SBATCH --job-name=RURHv',
                                            f'#SBATCH --output={out_path}',
                                            f'#SBATCH --error={out_path}',
                                            'cat /proc/sys/kernel/hostname',
                                            cmd]
                                            lines = [x+'\n' for x in lines]
                                            slurm_path = 'tmp.slurm'
                                            open(slurm_path,'w').writelines(lines)
                                            cmd = f'sbatch -w {nodes[c%len(nodes)]} {slurm_path}'
                                            
                                            if not go:
                                                print(lines[-1][:-1])
                                                exit()
                                            else:
                                                print(cmd)
                                                os.system(cmd)
                                        else:
                                            assert False
    print(c, cc)
