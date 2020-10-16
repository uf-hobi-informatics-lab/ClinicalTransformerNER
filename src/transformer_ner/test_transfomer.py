#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from transformer_ner.data_utils import TransformerNerDataProcessor, transformer_convert_data_to_features, ner_data_loader, batch_to_model_inputs, NEXT_TOKEN
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.tokenization_albert import AlbertTokenizer
from transformer_ner.model import RobertaNerModel, XLNetNerModel, AlbertNerModel
from transformer_ner.task import predict
import torch
from pathlib import Path

from pathlib import Path
from transformer_ner.transfomer_log import TransformerNERLogger


class Args:
    def __init__(self):
        self.model_type = 'bert'
        self.pretrained_model = 'mimicii_bert_10e_128b'
        # self.pretrained_model = 'bert-base-uncased'
        # self.model_type = 'roberta'
        # self.pretrained_model = 'roberta-base'
        # self.model_type = 'albert'
        # self.pretrained_model = 'albert-base-v2'
        # self.model_type = 'distilbert'
        # self.pretrained_model = 'distilbert-base-uncased'
        # self.model_type = 'xlnet'
        # self.pretrained_model = 'xlnet-base-cased'
        self.config_name = self.pretrained_model
        self.tokenizer_name = self.pretrained_model
        self.do_lower_case = True
        # self.data_dir = '/Users/alexgre/workspace/data/NER_data/i2b2_2010'
        self.data_dir = '/Users/alexgre/workspace/data/NER_data/2018n2c2/train_dev_drug'
        # self.data_dir = '/Users/alexgre/workspace/py3/pytorch_nlp/pytorch_UFHOBI_NER/test_data/conll-2003'
        self.data_has_offset_information = True
        self.new_model_dir = Path(__file__).resolve().parent.parent.parent / 'new_ner_model'
        self.predict_output_file = Path(__file__).resolve().parent.parent.parent / "new_ner_model/pred.txt"
        self.overwrite_output_dir = True
        self.max_seq_length = 64
        self.do_train = True
        self.do_predict = False
        self.model_selection_scoring = "strict-f_score-1"
        self.train_batch_size = 4
        self.eval_batch_size = 4
        self.learning_rate = 0.00001
        self.seed = 13
        self.logger = TransformerNERLogger(logger_level="i", logger_file=Path(__file__).resolve().parent.parent.parent/"new_ner_model/log.txt").get_logger()
        self.num_train_epochs = 1
        self.gradient_accumulation_steps = 1
        self.do_warmup = True
        self.label2idx = None
        self.idx2label = None
        self.max_num_checkpoints = 3
        self.warmup_ratio = 0.1
        self.weight_decay = 0.0
        self.adam_epsilon = 0.00000001
        self.max_grad_norm = 1.0
        self.log_file = None
        self.log_lvl = None
        self.fp16 = False
        self.local_rank = -1
        self.device = "cpu"
        self.train_steps = -1
        self.progress_bar = False
        self.early_stop = -1
        self.progress_bar = True
        self.save_model_core = True


def test():
    from pprint import pprint
    roberta_ner_data_processor = TransformerNerDataProcessor()
    conll_2003 = Path(__file__).resolve().parent.parent.parent / "test_data/conll-2003"
    roberta_ner_data_processor.set_data_dir(conll_2003)
    labels, label2idx = roberta_ner_data_processor.get_labels(default='roberta')
    print(labels, label2idx)

    # train_examples = roberta_ner_data_processor.get_train_examples()
    train_examples = roberta_ner_data_processor.get_test_examples()
    pprint(train_examples[:5], indent=1)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base_uncased")
    features = transformer_convert_data_to_features(train_examples[:5], label2idx, tokenizer, max_seq_len=10)

    model = RobertaNerModel.from_pretrained("roberta-base", num_labels=len(label2idx))
    # model = XLNetNerModel.from_pretrained("xlnet-base_uncased", num_labels=len(label2idx))

    y_trues, y_preds = [], []
    y_pred, y_true = [], []
    prev_gd = 0
    for idx, each_batch in enumerate(ner_data_loader(features, batch_size=5, task='test', auto=True)):
        # [idx*batch_size: (idx+1)*batch_size]
        print([(fea.input_tokens, fea.guards) for fea in features[idx*2: (idx+1)*2]])
        print(each_batch)

        original_tkid = each_batch[0].numpy()
        original_mask = each_batch[1].numpy()
        original_labels = each_batch[3].numpy()
        guards = each_batch[4].numpy()
        print(guards)

        inputs = batch_to_model_inputs(each_batch)

        with torch.no_grad():
            logits, flatted_logits, loss = model(**inputs)
            # get softmax output of the raw logits (keep dimensions)
            raw_logits = torch.argmax(torch.nn.functional.log_softmax(logits, dim=2), dim=2)
            raw_logits = raw_logits.detach().cpu().numpy()

        logits = logits.numpy()
        loss = loss.numpy()

        print(logits.shape)
        # print(loss)

        # tk=token, mk=mask, lb=label, lgt=logits
        for mks, lbs, lgts, gds in zip(original_mask, original_labels, raw_logits, guards):
            connect_sent_flag = False
            for mk, lb, lgt, gd in zip(mks, lbs, lgts, gds):
                if mk == 0:  # after hit first mask, we can stop for the current sentence since all rest will be pad
                    break
                if gd == 0 or prev_gd == gd:
                    continue
                if gd == -2:
                    connect_sent_flag = True
                    break
                if prev_gd != gd:
                    y_true.append(lb)
                    y_pred.append(lgt)
                    prev_gd = gd
            if connect_sent_flag:
                continue
            y_trues.append(y_true)
            y_preds.append(y_pred)
            y_pred, y_true = [], []
            prev_gd = 0
        print(y_trues)
        print(y_preds)


def test1():
    from transformer_ner.transfomer_log import TransformerNERLogger
    # logger = TransformerNERLogger(logger_file="/Users/alexgre/workspace/py3/pytorch_nlp/pytorch_UFHOBI_NER/log.txt").get_logger()
    logger = TransformerNERLogger().get_logger()
    print(logger)
    logger.info("hello world.")


def test2():
    args = Args()
    ner_data_processor = TransformerNerDataProcessor()
    conll_2003 = Path(__file__).resolve().parent.parent.parent / "test_data/conll-2003"
    ner_data_processor.set_data_dir(conll_2003)
    labels, label2idx = ner_data_processor.get_labels(default='roberta')
    # train_examples = roberta_ner_data_processor.get_train_examples()
    train_examples = ner_data_processor.get_test_examples()
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base_uncased")
    features = transformer_convert_data_to_features(args, train_examples[:5], label2idx, tokenizer, max_seq_len=10)

    model = AlbertNerModel.from_pretrained("albert-base-v2", num_labels=len(label2idx))

    for idx, each_batch in enumerate(ner_data_loader(features, batch_size=5, task='test', auto=True)):
        original_mask = each_batch[1].numpy()
        print(original_mask, original_mask.shape)
        inputs = batch_to_model_inputs(each_batch)
        with torch.no_grad():
            logits, flatted_logits, loss = model(**inputs)
        logits = logits.numpy()
        print(logits)
        print(logits.shape)
        break


if __name__ == '__main__':
    # test()
    # test1()
    test2()
