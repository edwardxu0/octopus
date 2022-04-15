import os
import copy
import time
import toml
import importlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from octopus.core.problem import Problem

from octopus.plot.box_plot import colored_box_plot


class Benchmark:
    def __init__(self, base_settings, study_settings, **kwargs):
        self.go = kwargs['go']
        self.slurm = kwargs['slurm']
        self.override = kwargs['override']
        self.slurm_config_path = kwargs['slurm_config_path']
        self.base_settings = toml.load(open(base_settings, 'r'))
        self.result_dir = os.path.join(kwargs['result_dir'], self.base_settings['name'])

        module = importlib.import_module(study_settings.split('.')[0].replace('/', '.'))
        for x in module.__dict__:
            if not x.startswith('__'):
                self.__setattr__(x, module.__dict__[x])
        self._define_problems()
        self._setup_()
    

    def _setup_(self):
        self.code_veri_answer = {'unsat': 1,
                            'sat': 2,
                            'unknown': 3,
                            'timeout': 4,
                            'memout': 4,
                            'error': 5}

        self.labels = ['artifact', 'network', 'heuristic', 'seed', 'property', 'epsilon', 'verifier', 'accuracy', 'stable relu', 'veri ans', 'veri time']

        self.tmp_dir = './tmp'
        self.sub_dirs = {}
        sub_dirs = {'train_log', 'model', 'property', 'figure', 'veri_log'}
        for sd in sub_dirs:
            sdp = os.path.join(self.result_dir, sd)
            self.sub_dirs[sd+'_dir'] = sdp


    def _define_problems(self):
        self.problems_t = []
        self.problems_v = []

        for a in self.artifacts:
            for n in self.networks:
                for h in self.heuristics:
                    for s in self.seeds:
                        self.problems_t += [(a,n,h,s)]
                        for p in self.props:
                            for e in self.epsilons:
                                for v in self.verifiers:
                                    self.problems_v += [(a,n,h,s,p,e,v)]


    def _exec(self, cmd, slurm_cmd):
        if not self.go:
            print('Dry run: ', cmd)
            exit()
        else:
            print('Fly: ', cmd)
            if not self.slurm:
                os.system(cmd)
            else:
                os.system(slurm_cmd)
                time.sleep(self.sleep_time)


    def _get_problem_paths(self, task, **kwargs):
        sts = copy.deepcopy(self.base_settings)

        a, n, h, s = list(kwargs.values())[:4]
        sts['train']['artifact'] = a
        sts['train']['net_name'] = n
        sts['train']['net_layers'] = self.networks[n]
        if h != 'base':
            sts['heuristic'][h] = self.heuristics[h]

        if task == 'T':
            log_name = f'{a}_{n}_{h}_{s}.txt'
            log_dir_postfix = 'train_log'
            config_dir_postfix = 'train_config'
            veri_log_path = None

        elif task == 'V':
            a, n, h, s, p, e, v = kwargs.values()
            sts['verify']['property'] = p
            sts['verify']['epsilon'] = e
            sts['verify']['verifier'] = v

            # configure log path
            log_name = f'{a}_{n}_{h}_{s}_{p}_{e}_{v}.txt'
            log_dir_postfix = 'veri_slurm_log'
            config_dir_postfix = 'veri_config'
            model_name = Problem.get_model_name(sts['train'], sts['heuristic'], s)
            veri_log_path = os.path.join(self.sub_dirs['veri_log_dir'], f"{model_name}_P={p}_E={e}_V={v}.txt")

        # dump octopus configurations
        log_dir = os.path.join(f"results/{sts['name']}", log_dir_postfix)
        config_dir = os.path.join(f"results/{sts['name']}", config_dir_postfix)
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        log_path = os.path.join(log_dir, log_name)
        Path(config_dir).mkdir(exist_ok=True, parents=True)
        toml_path = os.path.join(config_dir, f'{a}_{n}_{h}_{s}.toml')

        return sts, toml_path, log_path, veri_log_path
        

    def train(self):
        nb_done = 0
        nb_todo = 0
        for a, n, h, s in self.problems_t:

            sts, toml_path, train_log_path, _ =  self._get_problem_paths('T', a=a, n=n, h=h, s=s)
            
            # check if done
            if os.path.exists(train_log_path):
                nb_done += 1
                continue
            else:
                nb_todo += 1
            
            toml.dump(sts, open(toml_path, 'w'))
            cmd = f"python -m octopus {toml_path} train --seed {s}"
            
            slurm_cmd = None
            if self.slurm:
                lines = ['#!/bin/sh',
                        f'#SBATCH --job-name=O.T',
                        f'#SBATCH --output={train_log_path}',
                        f'#SBATCH --error={train_log_path}']
                if self.base_settings['train']['gpu']:
                    lines += [f'#SBATCH --partition=gpu',
                               '#SBATCH --gres=gpu:1']
                if self.train_nodes_ex:
                    lines += [f'#SBATCH --exclude={self.train_nodes_ex}']
                
                lines += ['cat /proc/sys/kernel/hostname',
                        'source .env.d/openenv.sh',
                        'echo $CUDA_VISIBLE_DEVICES',
                        cmd]

                lines = [x+'\n' for x in lines]
                open(self.slurm_config_path,'w').writelines(lines)

                
                assert not self.train_nodes
                slurm_cmd = f'sbatch  {self.slurm_config_path}'
            
            self._exec(cmd, slurm_cmd)
            
        print(f'Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_t)}.')


    def verify(self):
        nb_done = 0
        nb_todo = 0
        for i, (a, n, h, s, p, e, v) in enumerate(self.problems_v):
            sts, toml_path, veri_log_path, actual_veri_log_path =  self._get_problem_paths('V', a=a, n=n, h=h, s=s, p=p, e=e, v=v)

            # check if done
            if os.path.exists(veri_log_path) and os.path.exists(actual_veri_log_path):
                nb_done += 1
                continue
            else:
                print(veri_log_path)
                print(actual_veri_log_path)
                nb_todo += 1
            toml.dump(sts, open(toml_path, 'w'))
            
            cmd = f'python -m octopus {toml_path} verify --seed {s}'

            slurm_cmd = None
            if self.slurm:
                lines = ['#!/bin/sh',
                f'#SBATCH --job-name=O.V',
                f'#SBATCH --output={veri_log_path}',
                f'#SBATCH --error={veri_log_path}',
                'cat /proc/sys/kernel/hostname',
                'source .env.d/openenv.sh',
                cmd]
                lines = [x+'\n' for x in lines]
                
                open(self.slurm_config_path,'w').writelines(lines)

                param_node = f'-w {self.veri_nodes[i%len(self.veri_nodes)]}' if self.veri_nodes else ''
                slurm_cmd = f'sbatch {param_node} {self.slurm_config_path}'

            self._exec(cmd, slurm_cmd)

        print(f'Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_v)}.')


    # Analyze logs
    def analyze(self):
        # self.analyze_training()
        df_cache_path = os.path.join(self.result_dir, 'result.feather')
        if os.path.exists(df_cache_path) and not self.override:
            print('Using cached results.')
            df = pd.read_feather(df_cache_path)
        else:
            print('Parsing log files ...')
            df = self._parse_logs()
            df.to_feather(df_cache_path)
            print('Result cached.')
        
        self._analyze_training(df)
        self._analyze_verification(df)


    def _parse_logs(self):
        df = pd.DataFrame({x:[] for x in self.labels})
        print('Failed tasks:')
        print('--------------------')
        for i, (a, n, h, s, p, e, v) in enumerate(self.problems_v):
            _, _, train_log_path, _ =  self._get_problem_paths('T', a=a, n=n, h=h, s=s)
            lines = open(train_log_path, 'r').readlines()
            assert '[Test]' in lines[-4] and '[Train]' in lines[-5]
            accuracy = float(lines[-4].strip().split()[-1][:-2])
            stable_relu = int(lines[-5].strip().split()[-1])

            _, _, veri_log_path, actual_veri_log_path =  self._get_problem_paths('V', a=a, n=n, h=h, s=s, p=p, e=e, v=v)
            
            answer, time = Problem.analyze_veri_log(actual_veri_log_path)

            if answer is None or time is None:
                print('rm', veri_log_path)
                print('rm', actual_veri_log_path)
            
            df.loc[len(df.index)] = [a, n, h, s, p, e, v, accuracy, stable_relu, self.code_veri_answer[answer], time]
        print('--------------------')
        return df
        
            
    def _analyze_training(self, df):
        ...
    

    def _analyze_verification(self, df):
        
        print('Plotting training overview ...')
        # plot accuracy/stable relu among network/heuristic pairs.
        x_labels = []
        collection_accuracy = []
        collection_stable_relu = []
        for a in self.artifacts:
            for n in self.networks:
                for h in self.heuristics.keys():
                    dft = df
                    dft = dft[dft['artifact'] == a]
                    dft = dft[dft['network'] == n]
                    dft = dft[dft['heuristic'] == h]

                    acc = np.array(list(set(dft['accuracy'].values.tolist())))
                    stable_relu = np.array(list(set(dft['stable relu'].values.tolist())))

                    # TODO: make multiple plots for different artifacts
                    #x_labels += [f'{a} {n} {h}']
                    x_labels += [f'{n} {h}']
                    collection_accuracy += [acc]
                    collection_stable_relu += [stable_relu]
        
        collection_accuracy = np.array(collection_accuracy, dtype=object)
        collection_stable_relu = np.array(collection_stable_relu, dtype=object)

        fig, ax1 = plt.subplots(1, 1)
        ax2 = ax1.twinx()

        bp1 = colored_box_plot(ax1, collection_accuracy, 'red', 'tan')
        bp2 = colored_box_plot(ax2, collection_stable_relu, 'blue', 'cyan')
        
        plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Accuracy', 'Stable ReLU'], loc='upper left')

        ax1.set_xticklabels(x_labels*2, rotation=90)
        ax1.set_ylabel('Test Accuracy(%)')
        ax2.set_ylabel('# Stable ReLUs')
        ax1.set_xlabel('Network/Heuristics')
        plt.title('Test Accuracy/# Stable ReLUs vs. Heuristics and Networks pairs')
        plt.savefig(os.path.join(self.result_dir, 'Training_overview.pdf'), format="pdf", bbox_inches="tight")

        fig.clear()
        plt.close(fig)
        
        print('Plotting verification ...')
        for a in self.artifacts:
            for n in self.networks:
                for v in self.verifiers:
                    # plot verification time(Y) vs. Epsilon Values(X) for each heuristic
                    fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
                    ax2 = ax1.twinx()
                    title_prefix = f'[{n}:{v}]'
                    collection_verification_time = {}
                    collection_problem_solved = {}
                    for h in self.heuristics.keys():
                        avg_v_time = []
                        nb_solved = []
                        for e in self.epsilons:
                            dft = df
                            dft = dft[dft['artifact'] == a]
                            dft = dft[dft['network'] == n]
                            dft = dft[dft['verifier'] == v]
                            dft = dft[dft['heuristic'] == h]
                            dft = dft[dft['epsilon'] == e]
                            avg_v_time += [np.mean(dft['veri time'].to_numpy())]
                            nb_solved += [np.where(dft['veri ans'].to_numpy() < 3 )[0].shape[0]]

                        collection_verification_time[h] = avg_v_time
                        collection_problem_solved[h] = nb_solved
                        
                        ax1.plot(avg_v_time, label=h)
                        ax2.plot(nb_solved, linestyle='dashed')

                    ax1.legend(loc='center left')
                    ax1.set_xlabel('Epsilon')
                    ax1.set_ylabel('Verification Time(s)')
                    ax2.set_ylabel('Solved Problems')
                    ax1.set_ylim(-600*0.05, 600*1.05)
                    ax2.set_ylim(-1, 26)
                    ax1.set_xticks(range(len(self.epsilons)))
                    ax1.set_xticklabels(self.epsilons)
                    plt.title(title_prefix + ' Avg. Verification Time/Problems Solved vs. Epsilons')
                    plt.savefig(os.path.join(self.result_dir, f'Verification_{title_prefix}.pdf'), format="pdf", bbox_inches="tight")
                    fig.clear()
                    plt.close(fig)

