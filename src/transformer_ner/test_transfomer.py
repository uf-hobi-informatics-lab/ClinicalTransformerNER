#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

from transformer_ner.task import run_task
from transformer_ner.transfomer_log import TransformerNERLogger


class Args:
    def __init__(self, model_type, pretrained_model, do_train=True, do_predict=True,
                 new_model_dir=None, resume_from_model=None):
        self.model_type = model_type
        self.pretrained_model = pretrained_model if resume_from_model is None else resume_from_model
        self.config_name = self.pretrained_model
        self.tokenizer_name = self.pretrained_model
        self.do_lower_case = True
        self.overwrite_model_dir = True
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'test_data/conll-2003'
        self.data_has_offset_information = False
        self.new_model_dir = new_model_dir if new_model_dir is not None else Path(
            __file__).resolve().parent.parent.parent / f'new_ner_model/{model_type}_new_ner_model'
        self.predict_output_file = Path(
            __file__).resolve().parent.parent.parent / f"new_ner_model/{model_type}_new_ner_model/pred.txt"
        self.overwrite_output_dir = True
        self.max_seq_length = 16
        self.do_train = do_train
        self.do_predict = do_predict
        self.model_selection_scoring = "strict-f_score-1"
        self.train_batch_size = 4
        self.eval_batch_size = 4
        self.learning_rate = 0.00001
        self.min_lr = 1e-6
        self.seed = 13
        self.logger = TransformerNERLogger(
            logger_level="i",
            logger_file=Path(__file__).resolve().parent.parent.parent / "new_ner_model/log.txt").get_logger()
        self.num_train_epochs = 1
        self.gradient_accumulation_steps = 1
        self.do_warmup = True
        self.label2idx = None
        self.idx2label = None
        self.max_num_checkpoints = 1
        self.warmup_ratio = 0.1
        self.weight_decay = 0.0
        self.adam_epsilon = 0.00000001
        self.max_grad_norm = 1.0
        self.log_file = None
        self.log_lvl = None
        self.fp16 = False
        self.local_rank = -1
        self.device = "cpu"
        self.train_steps = 10
        self.early_stop = -1
        self.progress_bar = True
        self.save_model_core = True
        self.use_crf = False
        self.focal_loss = False
        self.focal_loss_gamma = 2
        self.resume_from_model = resume_from_model
        self.use_biaffine = False
        self.mlp_dim = 128
        self.mlp_layers = 0
        self.adversarial_training_method = None  # None, "fgm", "pgd"


def test():
    # test training
    for each in [
        ('bert', 'bert-base-uncased'),
        ('deberta-v2', "microsoft/deberta-xlarge-v2"),
        ("megatron",
         "/Users/alexgre/workspace/py3/HOBI-lab/models/345m_uf_full_deid_pubmed_mimic_wiki_fullcased50k_release"),
        ('deberta', "microsoft/deberta-base"),
        ('roberta', 'roberta-base'),
        ('xlnet', 'xlnet-base-cased')
    ]:
        args = Args(each[0], each[1], do_train=True, do_predict=False)
        run_task(args)


def test1():
    # test prediction
    args = Args("bert", 'bert-base-uncased', do_train=False, do_predict=True,
                new_model_dir=Path(
                    __file__).resolve().parent.parent.parent / "new_ner_model" / "bert-base-uncased_conll2003")
    run_task(args)


def test2():
    # test continuous training from existing NER model
    args = Args("bert", 'bert-base-uncased', do_train=True, do_predict=True,
                resume_from_model=Path(
                    __file__).resolve().parent.parent.parent / "new_ner_model" / "bert-base-uncased_conll2003")
    run_task(args)


def test3():
    # test prediction
    args = Args("bert", 'bert-base-uncased', do_train=True, do_predict=True,
                new_model_dir=Path(
                    __file__).resolve().parent.parent.parent / "new_ner_model" / "bert-base-uncased_conll2003")
    args.use_crf = True
    run_task(args)


if __name__ == '__main__':
    which_test = input("run which test? 1 or 2 or 3")
    if which_test == "0":
        test()
    elif which_test == "1":
        test1()
    elif which_test == "2":
        test2()
    elif which_test == "3":
        test3()
