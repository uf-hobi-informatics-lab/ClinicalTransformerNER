#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from transformer_ner.task import run_task
from transformer_ner.transfomer_log import TransformerNERLogger


class Args:
    def __init__(self, model_type, pretrained_model):
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.config_name = self.pretrained_model
        self.tokenizer_name = self.pretrained_model
        self.do_lower_case = True
        self.overwrite_model_dir = True
        self.data_dir = Path(__file__).resolve().parent.parent.parent/'test_data/conll-2003'
        self.data_has_offset_information = False
        self.new_model_dir = Path(__file__).resolve().parent.parent.parent/f'new_ner_model/{model_type}_new_ner_model'
        self.predict_output_file = Path(__file__).resolve().parent.parent.parent/f"new_ner_model/{model_type}_new_ner_model/pred.txt"
        self.overwrite_output_dir = True
        self.max_seq_length = 16
        self.do_train = True
        self.do_predict = True
        self.model_selection_scoring = "strict-f_score-1"
        self.train_batch_size = 4
        self.eval_batch_size = 4
        self.learning_rate = 0.00001
        self.seed = 13
        self.logger = TransformerNERLogger(
            logger_level="i",
            logger_file=Path(__file__).resolve().parent.parent.parent/"new_ner_model/log.txt").get_logger()
        self.num_train_epochs = 2
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
        self.train_steps = 100
        self.early_stop = -1
        self.progress_bar = True
        self.save_model_core = True
        self.use_crf = False


def test():
    for each in [('deberta', "microsoft/deberta-base"),
                 ('bert', 'bert-base-uncased'),
                 ('roberta', 'roberta-base'),
                 ('xlnet', 'xlnet-base-cased')]:
        args = Args(each[0], each[1])
        run_task(args)


if __name__ == '__main__':
    test()
