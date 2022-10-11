# !/usr/bin/python
#  -*- coding: utf-8 -*-

import logging
from pathlib import Path

from ..common_utils.common_log import create_logger


class TransformerNERLogger:
    def __init__(self, logger_file=None, logger_level=logging.DEBUG):
        self.lf = logger_file
        self.lvl = logger_level

    def set_log_info(self, logger_file, logger_level):
        self.lf = logger_file
        self.lvl = logger_level

    def get_logger(self):
        if self.lf:
            Path(self.lf).parent.mkdir(parents=True, exist_ok=True)
        return create_logger("Transformer_NER", log_level=self.lvl, set_file=self.lf)
