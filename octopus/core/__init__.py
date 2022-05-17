import datetime
from .problem import Problem


def workout(settings):
    problem = Problem(settings)

    start_time = datetime.datetime.now()
    if settings.task == "T":
        settings.logger.info("Training ...")
        problem.train()

    elif settings.task == "V":
        settings.logger.info("Verifying ...")
        problem.verify()

    elif settings.task == "A":
        settings.logger.info("Analyzing ...")
        problem.analyze()

    elif settings.task == "AA":
        settings.logger.info("Training ...")
        problem.train()
        settings.logger.info("Verifying ...")
        problem.verify()
        settings.logger.info("Analyzing ...")
        problem.analyze()
    else:
        raise NotImplementedError
    settings.logger.info("Mission Complete.")
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    settings.logger.info(f"Spent {duration:.2f} seconds.")
