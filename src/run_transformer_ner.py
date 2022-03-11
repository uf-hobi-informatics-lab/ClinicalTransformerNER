#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import warnings
import traceback

import torch
import transformers
from packaging import version

from transformer_ner.task import run_task
from transformer_biaffine_ner.task import run_task as run_biaffine_task
from transformer_ner.transfomer_log import TransformerNERLogger

pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'we now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)


def main():
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_type", default='bert', type=str, required=True,
                        help="valid values: bert, roberta or xlnet")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="The pretrained model file or directory for fine tuning.")
    # resume training on a NER model if set it will overwrite pretrained_model
    parser.add_argument("--resume_from_model", type=str, default=None,
                        help="The NER model file or directory for continuous fine tuning.")
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
                        help="Number of trianing steps between two evaluations on the dev set; "
                             "if <0 then evaluate after each epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--min_lr", default=1e-6, type=float,
                        help="The minimum number that lr can decay to.")
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
                        help="""The training will stop after num of epoch without performance improvement. 
                        If set to 0 or -1, then not use early stop.""")
    parser.add_argument('--focal_loss', action='store_true',
                        help="use focal loss function instead of cross entropy loss")
    parser.add_argument("--focal_loss_gamma", default=2, type=int,
                        help="the gamma hyperparameter in focal loss, commonly use 1 or 2")
    parser.add_argument("--use_biaffine", action='store_true',
                        help="Whether to use biaffine for NER (https://www.aclweb.org/anthology/2020.acl-main.577/).")
    parser.add_argument("--mlp_dim", default=128, type=int,
                        help="The output dimension for MLP layer in biaffine module, default to 128."
                             "If set this value <= 0, we use transformer model hidden layer dimension")
    parser.add_argument("--mlp_layers", default=0, type=int,
                        help="The number of layers in MLP in biaffine module, default to 0 (1 linear layer)."
                             "if set to 1, then NLP will have three linear layers")
    parser.add_argument("--mlp_hidden_dim", default=0, type=int,
                        help="The hidden dim of MLP layers in biaffine module, default to 0 (no use hidden layer)")
    # adversarial training method: pgd, fgm
    parser.add_argument("--adversarial_training_method", default=None,
                        help="what method to use for adversarial training, support pgd and fgm; "
                             "default is None which disable this function")
    # fp16 and distributed training (we use pytorch naive implementation instead of Apex)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    #  TODO: data parallel - support single node multi GPUs (use deepspeed only or pytorch naive ddp?)
    # parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    global_args = parser.parse_args()

    # create logger
    logger = TransformerNERLogger(global_args.log_file, global_args.log_lvl).get_logger()
    global_args.logger = logger

    # set and check cuda (we recommend to set up CUDA device in shell)
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_args.cuda_ids
    global_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Task will use cuda device: GPU_{}.".format(
        torch.cuda.current_device()) if torch.cuda.device_count() else 'Task will use CPU.')

    # if use resume_from_model, then resume_from_model will overwrite pretrained_model
    if global_args.resume_from_model:
        global_args.pretrained_model = global_args.resume_from_model

    if global_args.resume_from_model is None and global_args.pretrained_model is None:
        raise RuntimeError("""Both resume_from_model and pretrained_model are not set.
        You have to specify one of them.""")

    # if args.tokenizer_name and args.config_name are not specially set, set them as pretrained_model
    if not global_args.tokenizer_name:
        global_args.tokenizer_name = global_args.pretrained_model
        logger.warning("set tokenizer as {}".format(global_args.tokenizer_name))

    if not global_args.config_name:
        global_args.config_name = global_args.pretrained_model
        logger.warning("set config as {}".format(global_args.config_name))

    if global_args.do_predict and not global_args.predict_output_file:
        raise RuntimeError("Running prediction but predict output file is not set.")

    if global_args.focal_loss and global_args.use_crf:
        warnings.warn(
            "Using CRF cannot apply focal loss. CRF use viterbi decoding and loss will be calculated independently.")
        warnings.warn("We will overwrite focal loss to false and use CRF as default.")
        global_args.focal_loss = False

    if global_args.use_crf and global_args.use_biaffine:
        raise RuntimeError("You can not run both CRF and biaffine. Choose only one or None of them to proceed.")

    try:
        if global_args.use_biaffine:
            run_biaffine_task(global_args)
        else:
            run_task(global_args)
    except Exception as ex:
        traceback.print_exc()
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
