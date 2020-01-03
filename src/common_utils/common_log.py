#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

LOG_LVLs = {
    'i': logging.INFO,
    'd': logging.DEBUG,
    'e': logging.ERROR,
    'w': logging.WARN
}


def create_logger(logger_name="", log_level="d", set_file=None):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(LOG_LVLs[log_level])
    if set_file:
        fh = logging.FileHandler(set_file)
        fh.setFormatter(formatter)
        fh.setLevel(LOG_LVLs[log_level])
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(LOG_LVLs[log_level])
        logger.addHandler(ch)

    return logger
