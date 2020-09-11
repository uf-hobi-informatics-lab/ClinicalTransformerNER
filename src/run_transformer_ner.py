#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import torch
from transformer_ner.task import run_task
from transformer_ner.transfomer_log import TransformerNERLogger
from traceback import format_exc

from packaging import version
import transformers


pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'we now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)


def main():
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_type", default='bert', type=str, required=True,
                        help="valid values: bert, roberta or xlnet")
    parser.add_argument("--pretrained_model", type=str, required=True,
                        help="The pretrained model file or directory for fine tuning.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as pretrained_model")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as pretrained_model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--data_has_offset_information", action='store_true',
                        help="The input data directory.")
    parser.add_argument("--new_model_dir", type=str, required=True,
                        help="directory for saving new model checkpoints (keep latest n only)")
    parser.add_argument("--save_model_core", action='store_true',
                        help="""save the transformer core of the model 
                        which allows model to be used as base model for further pretraining""")
    parser.add_argument("--predict_output_file", type=str, default=None,
                        help="predicted results output file.")
    parser.add_argument('--overwrite_model_dir', action='store_true',
                        help="Overwrite the content of the new model directory")
    parser.add_argument("--seed", default=3, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--model_selection_scoring", default='strict-f_score-1', type=str,
                        help="""The scoring methos used to select model on dev dataset 
                        only support strict-f_score-n, relax-f_score-n (n is 0.5, 1, or 2)""")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--use_crf", action='store_true',
                        help="Whether to use crf layer as classifier.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="The batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="The batch size for eval.")
    parser.add_argument('--train_steps', type=int, default=-1,
                        help="Number of trianing steps between two evaluations on the dev set; if <0 then evaluate after each epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_warmup", action='store_true',
                        help='Whether to apply warmup strategy in optimizer.')
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_num_checkpoints", default=3, type=int,
                        help="max number of checkpoints saved during training, old checkpoints will be removed.")
    parser.add_argument("--log_file", default=None,
                        help="where to save the log information")
    parser.add_argument("--log_lvl", default="i", type=str,
                        help="d=DEBUG; i=INFO; w=WARNING; e=ERROR")
    parser.add_argument("--progress_bar", action='store_true',
                        help="show progress during the training in tqdm")
    parser.add_argument("--early_stop", default=-1, type=int,
                        help="""The training will stop after num of epoch without performance improvement. If set to 0 or -1, then not use early stop.""")

    # fp16 and distributed training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument("--fp16_opt_level", type=str, default="O1",
    #                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #                          "See details at https://nvidia.github.io/apex/amp.html")
    # parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    # parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    global_args = parser.parse_args()

    # create logger
    logger = TransformerNERLogger(global_args.log_file, global_args.log_lvl).get_logger()
    global_args.logger = logger

    # set and check cuda (we recommend to set up CUDA device in shell)
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_args.cuda_ids
    global_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Task will use cuda device: GPU_{}.".format(torch.cuda.current_device()) if torch.cuda.device_count() else 'Task will use CPU.')

    # if args.tokenizer_name and args.config_name are not specially set, set them as pretrained_model
    if not global_args.tokenizer_name:
        global_args.tokenizer_name = global_args.pretrained_model
        logger.warning("set tokenizer as {}".format(global_args.tokenizer_name))

    if not global_args.config_name:
        global_args.config_name = global_args.pretrained_model
        logger.warning("set config as {}".format(global_args.config_name))

    if global_args.do_predict and not global_args.predict_output_file:
        raise RuntimeError("Running prediction but predict output file is not set.")

    try:
        run_task(global_args)
    except Exception as ex:
        logger.error(format_exc())


if __name__ == '__main__':
    main()
