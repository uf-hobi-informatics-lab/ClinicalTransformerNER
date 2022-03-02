# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

from transformer_biaffine_ner.task_utils import train, predict


def run_task(args):
    args.logger.info("Training with Biaffine mode...")

    if args.do_train:
        pass