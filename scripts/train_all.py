import os

from run import TRAIN_LOG_DIR
from src.configs import _model_name


def train_all(strat, problems, nets, itst, ocrc, upbd, seeds, slurm=False, go=False):
    c = 0
    for p in problems:
        for n in nets:
            for i in itst:
                for j in ocrc:
                    for k in upbd:
                        for s in seeds:
                            c += 1
                            cmd = f"python src/main.py train --epochs 40 --save-model --log-interval 1 --problem {p} --net-name {n} --seed {s}"
                            
                            if strat == 'rurh':
                                cmd += f" --rurh ral --rurh_itst {i} --rurh_ocrc {j} --rurh_upbd {k} --rurh_ral 0.05 --rurh_deactive_pre 10 --rurh_deactive_post 10"
                            elif strat == 'rs':
                                cmd += f" --rs basic"
                            elif strat == 'baseline':
                                pass
                            else:
                                assert False

                            m_name = _model_name(strat, p, n, s, rurh_itst=i, rurh_ocrc=j, rurh_upbd=k)

                            out_path = f'{TRAIN_LOG_DIR}/{m_name}.txt'
                            if os.path.exists(out_path):
                                print('Done.')
                                continue
                            if slurm:
                                lines = ['#!/bin/sh',
                                        f'#SBATCH --job-name=RURHt',
                                        #f'#SBATCH --gres=gpu:1',
                                        f'#SBATCH --output={out_path}',
                                        f'#SBATCH --error={out_path}',
                                        'cat /proc/sys/kernel/hostname',
                                        cmd]

                                lines = [x+'\n' for x in lines]
                                slurm_path = 'tmp.slurm'
                                open(slurm_path,'w').writelines(lines)
                                cmd = f'sbatch -c 12 {slurm_path}'
                                #cmd = f'sbatch --partition=gpu {slurm_path}'
                                
                                if not go:
                                    print(f'{lines[-1][:-1]} > {out_path}')
                                    exit()
                                else:
                                    print(cmd)
                                    os.system(cmd)
                                # time.sleep(1)
                            else:
                                assert False
    print(f'# tasks: {c}')
