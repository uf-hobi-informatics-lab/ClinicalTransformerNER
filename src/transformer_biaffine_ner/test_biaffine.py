# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 3/4/22
import sys
from pathlib import Path

from transformer_biaffine_ner.task import run_task
from transformer_ner.transfomer_log import TransformerNERLogger


class Args:
    def __init__(self, model_type, pretrained_model, do_train=True, do_predict=True):
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.config_name = self.pretrained_model
        self.tokenizer_name = self.pretrained_model
        self.do_lower_case = True
        self.overwrite_model_dir = True
        self.data_dir = (Path(__file__).resolve().parent.parent.parent / 'test_data/biaffine_conll2003_mini')\
            .as_posix()
        self.data_has_offset_information = False
        self.new_model_dir = (
                Path(__file__).resolve().parent.parent.parent / f'new_ner_model/{model_type}_biaffine_ner_model')\
            .as_posix()
        self.predict_output_file = (
                Path(__file__).resolve().parent.parent.parent / f"new_ner_model/{model_type}_biaffine_ner_model/pred.json")\
            .as_posix()
        self.overwrite_output_dir = True
        self.max_seq_length = 512
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
        self.train_steps = 10
        self.early_stop = 3
        self.progress_bar = True
        self.save_model_core = False
        self.use_crf = False
        self.focal_loss = False
        self.focal_loss_gamma = 2
        self.resume_from_model = None
        self.use_biaffine = True
        self.mlp_dim = 128
        self.mlp_layers = 0
        self.mlp_hidden_dim = 0
        self.adversarial_training_method = None  # None, "fgm", "pgd"


def test():
    args = Args("bert", "bert-base-uncased", do_train=True, do_predict=False)
    run_task(args)


def test1():
    args = Args("bert", "bert-base-uncased", do_train=False, do_predict=True)
    run_task(args)


if __name__ == '__main__':
    which_test = input("run which test? 0 or 1")
    if which_test == "0":
        test()
    elif which_test == "1":
        test1()
