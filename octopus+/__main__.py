#!/usr/bin/env python

import argparse

from pyfiglet import Figlet
from datetime import datetime

from .benchmark import Benchmark


def _parse_args():
    parser = argparse.ArgumentParser(description='OCTOPUS Benchmarks.', prog='octopus+')
    parser.add_argument('base_settings', type=str,
                        help='An OCTOPUS toml config file as bases.')
    parser.add_argument('study_settings', type=str,
                        help='A python file containing the parameters to explore.')
    parser.add_argument('task', type=str,choices=['T','V','A'],
                        help='Tasks to perform on the benchmark, including [T]rain, [V]erify, and [A]nalyze.')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Root result directory.')
    parser.add_argument('--slurm_config_path', type=str, default='tmp/tmp.slurm',
                        help='Path to slurm config temp file.')
    parser.add_argument('--override', action='store_true',
                        help='Overrides existing train/verify/analyze tasks.')
    parser.add_argument('--slurm', action='store_true',
                        help='Run on SLURM or not?')
    parser.add_argument('--go', action='store_true',
                        help='Dry run or not?')

    return parser.parse_args()


def main():
    f = Figlet(font='slant')
    print(f.renderText('OCTOPUS +'), end='')

    time_start = datetime.now()
    args = _parse_args()
    study = Benchmark(args.base_settings, args.study_settings,
                    go = args.go,
                    slurm = args.slurm,
                    override = args.override,
                    slurm_config_path = args.slurm_config_path,
                    result_dir = args.result_dir)

    if args.task == 'T':
        study.train()

    elif args.task == 'V':
        study.verify()

    elif args.task == 'A':
        study.analyze()
    else:
        assert False
    
    time_end = datetime.now()
    duration = time_end - time_start
    print(f'Spent {duration.seconds} seconds.')


if __name__ == '__main__':
    main()
