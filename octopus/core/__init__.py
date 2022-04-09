import argparse

from .problem import Problem


def workout(settings):
    problem = Problem(settings)

    if settings.task == 'train':
        problem.train()

    elif settings.task == 'verify':
        problem.verify()

    elif settings.task == 'analyze':
        problem.analyze()

    elif settings.task == 'all':
        problem.train()
        problem.verify()
        problem.analyze()

    else:
        raise NotImplementedError