import os
import copy
import toml
from pathlib import Path

EX_NODE = 'affogato12,affogato13,affogato14,cheetah01,ristretto01,ristretto02,ristretto03,ristretto04,lynx08,lynx09,ai04'


def train_all(settings, artifacts, nets, heuristics, seeds, slurm=False, go=False):
    c = 0
    for a in artifacts:
        for n in nets:
            for h in heuristics:
                for s in seeds:
                    c += 1
                    sts = copy.deepcopy(settings)
                    sts['train']['artifact'] = a
                    sts['train']['net_name'] = n
                    sts['train']['net_layers'] = nets[n]
                    if h != 'base':
                        sts['heuristic'][h] = heuristics[h]
                    
                    train_config_dir = os.path.join(f"results/{sts['name']}", 'train_config')
                    Path(train_config_dir).mkdir(exist_ok=True, parents=True)
                    toml_path = os.path.join(train_config_dir, f'{a}_{n}_{h}_{s}.toml')
                    toml.dump(sts, open(toml_path, 'w'))

                    train_log_dir = os.path.join(f"results/{sts['name']}", 'train_log')
                    Path(train_log_dir).mkdir(exist_ok=True, parents=True)
                    train_log_path = os.path.join(train_log_dir, f'{a}_{n}_{h}_{s}.txt')
                
                    cmd = f"octopus {toml_path} train --seed {s}"

                    if os.path.exists(train_log_path):
                        print('Done.')
                        continue
                    if slurm:
                        lines = ['#!/bin/sh',
                                f'#SBATCH --job-name=OCTt',
                                f'#SBATCH --gres=gpu:1',
                                f'#SBATCH --output={train_log_path}',
                                f'#SBATCH --error={train_log_path}',
                                f'#SBATCH --exclude={EX_NODE}',
                                'cat /proc/sys/kernel/hostname',
                                'source .env.d/openenv.sh',
                                'echo $CUDA_VISIBLE_DEVICES',
                                cmd]

                        lines = [x+'\n' for x in lines]
                        slurm_path = 'tmp/tmp.slurm'
                        open(slurm_path,'w').writelines(lines)
                        #cmd = f'sbatch {slurm_path}'
                        cmd = f'sbatch  --partition=gpu {slurm_path}'
                        
                        if not go:
                            print(f'{lines[-1][:-1]}')
                            exit()
                        else:
                            print(cmd)
                            os.system(cmd)
                        # time.sleep(1)
                    else:
                        assert False
    print(f'# tasks: {c}')
