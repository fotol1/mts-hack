import argparse
from datetime import datetime
from functools import partial
import os
import time

from loguru import logger
import numpy as np
import pandas as pd


def createParser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions_path", type=str, default="Data/train_1.csv")

    return parser


if __name__ == "__main__":

    parser = createParser()
    args, _ = parser.parse_known_args()

    logfile = os.path.join("logs", f"{str(round(time.time()))}.log")

    logger.add(logfile)
    logger.info("Готовим файл из dwh для обучения моделей первого уровня")
