import argparse
from dataclasses import dataclass
import datetime
from .problem import Problem


def workout(settings):
    problem = Problem(settings)

    start_time = datetime.datetime.now()
    if settings.task == 'train':
        settings.logger.info('Training ...')
        problem.train()
        settings.logger.info('Mission Complete.')

    elif settings.task == 'verify':
        settings.logger.info('Verifying ...')
        problem.verify()
        settings.logger.info('Mission Complete.')

    elif settings.task == 'analyze':
        settings.logger.info('Analyzing ...')
        problem.analyze()
        settings.logger.info('Mission Complete.')

    elif settings.task == 'all':
        settings.logger.info('Training ...')
        problem.train()
        settings.logger.info('Verifying ...')
        problem.verify()
        settings.logger.info('Analyzing ...')
        problem.analyze()
        settings.logger.info('Mission Complete.')

    else:
        raise NotImplementedError

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    settings.logger.info(f'Spent {duration:.2f} seconds.')
