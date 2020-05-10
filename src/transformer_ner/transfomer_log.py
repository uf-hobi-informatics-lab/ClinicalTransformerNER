# !/usr/bin/python
#  -*- coding: utf-8 -*-

from common_utils.common_log import create_logger
import logging


class TransformerNERLogger:
    def __init__(self, logger_file=None, logger_level=logging.DEBUG):
        self.lf = logger_file
        self.lvl = logger_level

    def set_log_info(self, logger_file, logger_level):
        self.lf = logger_file
        self.lvl = logger_level

    def get_logger(self):
        return create_logger("Transformer_NER", log_level=self.lvl, set_file=self.lf)
