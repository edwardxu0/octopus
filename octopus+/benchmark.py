import os
import copy
import time
import toml
import uuid
import importlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from octopus.core.problem import Problem

from octopus.plot.box_plot import colored_box_plot


class Benchmark:
    def __init__(self, base_settings, benchmark_settings, **kwargs):
        self.go = kwargs['go']
        self.slurm = kwargs['slurm']
        self.override = kwargs['override']
        self.logger = kwargs['logger']
        self.base_settings = toml.load(open(base_settings, 'r'))
        self.result_dir = os.path.join(kwargs['result_dir'], self.base_settings['name'])

        self._setup_(benchmark_settings)
        self._define_problems()

    def _setup_(self, benchmark_settings):
        self.code_veri_answer = {'unsat': 1,
                                 'sat': 2,
                                 'unknown': 3,
                                 'timeout': 4,
                                 'memout': 4,
                                 'error': 5}

        self.labels = ['artifact', 'network', 'heuristic', 'seed', 'property',
                       'epsilon', 'verifier', 'accuracy', 'stable relu', 'veri ans', 'veri time']

        self.sub_dirs = {}
        sub_dirs = ['train_config', 'train_log', 'model', 'veri_config', 'veri_log', 'property', 'figure', ]

        if self.slurm:
            sub_dirs += ['train_slurm', 'veri_slurm']
            if 'save_log' in self.base_settings['train']:
                self.logger.info('Disabled [save_log] and redirected training logs to slurm logging.')
                self.base_settings['train']['save_log'] = False
            if 'save_log' in self.base_settings['verify']:
                self.logger.info('Disabled [save_log] and redirected verification logs to slurm logging.')
                self.base_settings['verify']['save_log'] = False

        for sd in sub_dirs:
            sdp = os.path.join(self.result_dir, sd)
            Path(sdp).mkdir(exist_ok=True, parents=True)
            self.sub_dirs[sd+'_dir'] = sdp

        self.logger.info('Reading benchmark settings ...')
        module = importlib.import_module(benchmark_settings.split('.')[0].replace('/', '.'))
        for x in module.__dict__:
            if not x.startswith('__'):
                self.__setattr__(x, module.__dict__[x])

    def _define_problems(self):
        self.logger.info('Configuring problems ...')
        self.problems_T = []
        self.problems_V = []

        for a in self.artifacts:
            for n in self.networks:
                for h in self.heuristics:
                    for s in self.seeds:
                        self.problems_T += [(a, n, h, s)]
                        for p in self.props:
                            for e in self.epsilons:
                                for v in self.verifiers:
                                    self.problems_V += [(a, n, h, s, p, e, v)]

    def _exec(self, cmd, slurm_cmd):
        if not self.go:
            self.logger.info(f'Dry run: {cmd}')
            exit(0)
        else:
            self.logger.info(f'Fly: {cmd}')
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
        sts['heuristic'] = {}
        if h != 'base':
            sts['heuristic'][h] = self.heuristics[h]

        model_name = Problem.get_model_name(sts['train'], sts['heuristic'], s)
        if task == 'T':
            log_path = os.path.join(self.sub_dirs['train_log_dir'], f"{model_name}.txt")
            config_path = os.path.join(self.sub_dirs['train_config_dir'], f'{model_name}.toml')
            slurm_script_path = None if not self.slurm else os.path.join(self.sub_dirs['train_slurm_dir'], f"{model_name}.slurm")

        elif task == 'V':
            a, n, h, s, p, e, v = kwargs.values()
            sts['verify']['property'] = p
            sts['verify']['epsilon'] = e
            sts['verify']['verifier'] = v

            # configure log path
            log_path = os.path.join(self.sub_dirs['veri_log_dir'], f"{model_name}_P={p}_E={e}_V={v}.txt")
            config_path = os.path.join(self.sub_dirs['veri_config_dir'], f"{model_name}_P={p}_E={e}_V={v}.toml")
            slurm_script_path = None if not self.slurm else os.path.join(self.sub_dirs['veri_slurm_dir'], f"{model_name}_P={p}_E={e}_V={v}.slurm")

        # dump octopus configurations
        return sts, config_path, slurm_script_path, log_path

    def train(self):
        self.logger.info('Training ...')
        nb_done = 0
        nb_todo = 0
        for i, (a, n, h, s) in enumerate(self.problems_T):

            sts, config_path, slurm_script_path, log_path = self._get_problem_paths('T', a=a, n=n, h=h, s=s)

            # check if done
            if os.path.exists(log_path) and not self.override:
                nb_done += 1
                continue
            else:
                nb_todo += 1

            toml.dump(sts, open(config_path, 'w'))
            cmd = f"python -m octopus {config_path} T --seed {s}"

            slurm_cmd = None
            if self.slurm:
                lines = ['#!/bin/sh',
                         f'#SBATCH --job-name=O.T',
                         f'#SBATCH --output={log_path}',
                         f'#SBATCH --error={log_path}']
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
                open(slurm_script_path, 'w').writelines(lines)
                param_node = f'-w {self.train_nodes[i%len(self.train_nodes)]}' if self.train_nodes else ''
                slurm_cmd = f'sbatch {param_node} {slurm_script_path}'

            self._exec(cmd, slurm_cmd)

        self.logger.info(f'Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_T)}.')

    def verify(self):
        self.logger.info('Verifying ...')
        nb_done = 0
        nb_todo = 0
        for i, (a, n, h, s, p, e, v) in enumerate(self.problems_V):
            sts, config_path, slurm_script_path, log_path = self._get_problem_paths(
                'V', a=a, n=n, h=h, s=s, p=p, e=e, v=v)

            # check if done
            if os.path.exists(log_path) and not self.override:
                nb_done += 1
                continue
            else:
                nb_todo += 1
            toml.dump(sts, open(config_path, 'w'))

            cmd = f'python -m octopus {config_path} V --seed {s}'

            tmpdir = f'/tmp/{uuid.uuid1()}'
            slurm_cmd = None
            if self.slurm:
                lines = ['#!/bin/sh',
                         f'#SBATCH --job-name=O.V',
                         f'#SBATCH --output={log_path}',
                         f'#SBATCH --error={log_path}',
                         f'export TMPDIR={tmpdir}',
                         f'mkdir {tmpdir}',
                         'cat /proc/sys/kernel/hostname',
                         'source .env.d/openenv.sh',
                         cmd,
                         f'rm -rf {tmpdir}', ]
                lines = [x+'\n' for x in lines]

                open(slurm_script_path, 'w').writelines(lines)
                param_node = f'-w {self.veri_nodes[i%len(self.veri_nodes)]}' if self.veri_nodes else ''
                slurm_cmd = f'sbatch {param_node} {slurm_script_path}'

            self._exec(cmd, slurm_cmd)

        self.logger.info(f'Tasks: done: {nb_done}, todo: {nb_todo}, total: {len(self.problems_V)}.')

    # Analyze logs
    def analyze(self):
        self.logger.info('Analyzing ...')
        # self.analyze_training()
        df_cache_path = os.path.join(self.result_dir, 'result.feather')
        if os.path.exists(df_cache_path) and not self.override:
            self.logger.info('Using cached results.')
            df = pd.read_feather(df_cache_path)
        else:
            self.logger.info('Parsing log files ...')
            df = self._parse_logs()
            df.to_feather(df_cache_path)
            self.logger.info('Result cached.')

        self._analyze_training(df)
        self._analyze_verification(df)

    def _parse_logs(self):
        df = pd.DataFrame({x: [] for x in self.labels})
        self.logger.info('Failed tasks:')
        self.logger.info('--------------------')
        for i, (a, n, h, s, p, e, v) in enumerate(self.problems_V):
            _, _, _, train_log_path = self._get_problem_paths('T', a=a, n=n, h=h, s=s)
            lines = open(train_log_path, 'r').readlines()
            assert '[Test]' in lines[-4]
            accuracy = float(lines[-4].strip().split()[-1][:-2])
            stable_relu = int(lines[-4].strip().split()[-1])

            _, _, _, veri_log_path = self._get_problem_paths('V', a=a, n=n, h=h, s=s, p=p, e=e, v=v)
            answer, time = Problem.analyze_veri_log(veri_log_path)
            if answer is None or time is None:
                print('rm', veri_log_path)

            df.loc[len(df.index)] = [a, n, h, s, p, e, v, accuracy, stable_relu, self.code_veri_answer[answer], time]
        self.logger.info('--------------------')
        return df

    def _analyze_training(self, df):
        ...

    def _analyze_verification(self, df):

        self.logger.info('Plotting training overview ...')
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

        self.logger.info('Plotting verification ...')
        for a in self.artifacts:
            for n in self.networks:
                for v in self.verifiers:
                    # plot verification time(Y) vs. Epsilon Values(X) for each heuristic
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
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
                            nb_solved += [np.where(dft['veri ans'].to_numpy() < 3)[0].shape[0]]

                        collection_verification_time[h] = avg_v_time
                        collection_problem_solved[h] = nb_solved

                        ax1.plot(avg_v_time, label=h)
                        ax2.plot(nb_solved, linestyle='dashed')

                    ax1.legend(loc='center left')
                    ax1.set_xlabel('Epsilon')
                    ax1.set_ylabel('Verification Time(s)')
                    ax2.set_ylabel('Solved Problems')
                    ax1.set_ylim(-self.base_settings['verify']['time']*0.05, self.base_settings['verify']['time']*1.05)
                    ax2.set_ylim(-1, 26)
                    ax1.set_xticks(range(len(self.epsilons)))
                    ax1.set_xticklabels(self.epsilons)
                    plt.title(title_prefix + ' Avg. Verification Time/Problems Solved vs. Epsilons')
                    plt.savefig(os.path.join(self.result_dir,
                                f'Verification_{title_prefix}.pdf'), format="pdf", bbox_inches="tight")
                    fig.clear()
                    plt.close(fig)
