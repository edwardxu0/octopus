import os
import toml
import copy
import time

from pathlib import Path


def verify_all(settings, artifact, nets, heuristics, seeds, props, epsilons, verifiers, slurm=False, wb=False, go=False):
    nodes = ['cortado'+x for x in ['01', '02','03','04','05','06','07','08','09','10']]

    c = -1
    cc = 0
    for a in artifact:
        for n in nets:
            for h in heuristics:
                for s in seeds:
                    for p in props:
                        for e in epsilons:
                            for v in verifiers:
                                c += 1
                                sts = copy.deepcopy(settings)
                                sts['train']['artifact'] = a
                                sts['train']['net_name'] = n
                                sts['train']['net_layers'] = nets[n]
                                if h != 'base':
                                    sts['heuristic'][h] = heuristics[h]
                                sts['verify']['property'] = p
                                sts['verify']['epsilon'] = e
                                sts['verify']['verifier'] = v

                                veri_config_dir = os.path.join(f"results/{sts['name']}", 'veri_config')
                                Path(veri_config_dir).mkdir(exist_ok=True, parents=True)
                                toml_path = os.path.join(veri_config_dir, f'{a}_{n}_{h}_{s}_{p}_{e}_{v}.toml')
                                toml.dump(sts, open(toml_path, 'w'))

                                veri_slurm_dir = os.path.join(f"results/{sts['name']}", 'veri_slurm_log')
                                Path(veri_slurm_dir).mkdir(exist_ok=True, parents=True)
                                veri_log_path = os.path.join(veri_slurm_dir, f'{a}_{n}_{h}_{s}_{p}_{e}_{v}.txt')
                                cmd = f'octopus {toml_path} verify --seed {s}'

                                if os.path.exists(veri_log_path):
                                    continue
                                if slurm:
                                    lines = ['#!/bin/sh',
                                    f'#SBATCH --job-name=RURHv',
                                    f'#SBATCH --output={veri_log_path}',
                                    f'#SBATCH --error={veri_log_path}',
                                    'cat /proc/sys/kernel/hostname',
                                    'source .env.d/openenv.sh',
                                    cmd]
                                    lines = [x+'\n' for x in lines]
                                    slurm_path = 'tmp/tmp.slurm'
                                    open(slurm_path,'w').writelines(lines)
                                    cmd = f'sbatch -w {nodes[c%len(nodes)]} {slurm_path}'
                                    
                                    if not go:
                                        print(lines[-1][:-1])
                                        exit()
                                    else:
                                        print(cmd)
                                        time.sleep(1)
                                        os.system(cmd)
                                else:
                                    assert False
    print(c, cc)
