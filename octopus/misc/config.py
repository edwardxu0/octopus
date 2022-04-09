import os
import toml
import pathlib
import logging


def configure(args):
    config_file = open(args.configs, 'r').read()
    cfg = toml.loads(config_file)
    cfg['result_dir'] = os.path.join(args.result_dir, f'{cfg["name"]}.{args.seed}')
    cfg['seed'] = args.seed
    cfg['task'] = args.task

    if args.debug:
        cfg['logging_level'] = logging.DEBUG
    elif args.dumb:
        cfg['logging_level'] = logging.WARN
    else:
        cfg['logging_level'] = logging.INFO

    settings = Settings(cfg)
    return settings


class Settings():
    def __init__(self, cfg):
        self.name = cfg['name']
        self.logging_level = cfg['logging_level']
        self.result_dir = cfg['result_dir']
        self.seed = cfg['seed']
        self.task = cfg['task']

        self.cfg_train = cfg['train']
        self.cfg_relu_est = cfg['ReLU_estimation']
        self.cfg_heuristic = cfg['heuristic'] if 'heuristic' in cfg else None
        self.cfg_verify = cfg['verify']

        self.answer_code = {'unsat': 1,
                            'sat': 2,
                            'unknown': 3,
                            'timeout': 4,
                            'memout': 4,
                            'error': 5}

        self.tmp_dir = './tmp'
        self.sub_dirs = ['train_log', 'model', 'property', 'veri_log']
        if cfg['train']['dispatch']['platform'] == 'slurm':
            self.sub_dirs += ['train_slurm']
        if cfg['verify']['dispatch']['platform'] == 'slurm':
            self.sub_dirs += ['veri_slurm']
        self._make_dirs()
        cfg['result_dir']


    def _make_dirs(self):
        pathlib.Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path().mkdir(parents=True, exist_ok=True)
        for sd in self.sub_dirs:
            attr = sd + '_dir'
            self.__setattr__(attr, os.path.join(self.result_dir, sd))
            pathlib.Path(self.__getattribute__(attr)).mkdir(
                parents=True, exist_ok=True)
        
