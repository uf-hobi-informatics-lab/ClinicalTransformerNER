#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AlbertTokenizer, BartTokenizer, BertTokenizer,
                          DebertaTokenizer, DistilBertTokenizer,
                          ElectraTokenizer, LongformerTokenizer,
                          RobertaTokenizer, XLNetTokenizer, DebertaV2Tokenizer)
from NLPreprocessing.annotation2BIO import pre_processing, generate_BIO
from NLPreprocessing.text_process.sentence_tokenization import SentenceBoundaryDetection
NEXT_TOKEN = "[next]"
NEXT_GUARD = -2


class InputExample(object):
    def __init__(self, guide, text, label=None, offsets=None):
        self.guide = guide
        self.text = text
        self.label = label
        self.offsets = offsets

    def __repr__(self):
        return str(self.__dict__)


class InputFeature(object):
    def __init__(self, input_tokens, input_ids, attention_mask, segment_ids, label, guards=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = segment_ids
        self.label = label
        self.guards = guards

    def __repr__(self):
        return str(self.__dict__)


class TransformerNerDataProcessor(object):
    def __init__(self):
        self.data_dir = None
        self.has_offset_info = False
        self.logger = None
        self.sent_tokenizer = SentenceBoundaryDetection()

    def set_logger(self, logger):
        self.logger = logger

    def set_data_dir(self, data_dir):
        self.data_dir = Path(data_dir)

    def offset_info_available(self):
        self.has_offset_info = True

    def get_train_examples(self, file_name=None):
        input_file_name = self.data_dir / file_name if file_name else self.data_dir / "train.txt"
        data, _ = self._read_data(input_file_name, task='train')
        return self._create_examples(data, 'train')

    def get_dev_examples(self, file_name=None):
        input_file_name = self.data_dir / file_name if file_name else self.data_dir / "dev.txt"
        data, _ = self._read_data(input_file_name, task='train')
        return self._create_examples(data, 'dev')

    def get_test_examples(self, file_name=None, use_bio=True, sents=None):
        input_file_name = self.data_dir / file_name if file_name else self.data_dir / "test.txt"
        data, _ = self._read_data(input_file_name, task='test', use_bio=use_bio, sents=sents)
        return self._create_examples(data, 'test')

    def get_labels(self, default='bert', customized_label2idx=None):
        """
        X will be used in tokenization to fill extra-generated tokens from original tokens
        default is used to initialized the label2idx dict; will support bert, roberta and xlnet
        the system labels must be smaller than label id for 'O', otherwise there will be errors in prediction
        """
        if default in {'bert', 'megatron', 'roberta', 'albert',
                       'distilbert', 'bart', 'xlnet',
                       'electra', 'deberta', 'longformer',
                       'deberta-v2'} and not customized_label2idx:
            # we do not need special label for SEP, using O instead
            # label2idx = {'O': 4, 'X': 3, 'PAD': 0, 'CLS': 1, 'SEP': 2}
            label2idx = {'O': 3, 'X': 2, 'PAD': 0, 'CLS': 1}
        else:
            if customized_label2idx:
                self.logger.warning('''The customized label2idx may not be compatible with the output check; 
Make sure all the control labels' indexes are smaller than the index of label O;
Otherwise will cause prediction error.''')
                label2idx = customized_label2idx
            else:
                raise ValueError('The default parameter {} as a model type is not supported '.format(default))

        _, train_labels = self._read_data(self.data_dir / "train.txt", task='train')
        _, dev_labels = self._read_data(self.data_dir / "dev.txt", task='train')

        if dev_labels.intersection(train_labels) != dev_labels:
            self.logger.warning("dev set has label ({}) not appeared in train set.".format(
                {e for e in dev_labels if e not in train_labels}))

        for lb in sorted(train_labels, key=lambda x: x.split("-")[-1]):
            if lb not in label2idx:
                label2idx[lb] = len(label2idx)

        return list(label2idx.keys()), label2idx

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, offsets, label) in enumerate(lines):
            guide = "{}-{}".format(set_type, i)
            examples.append(InputExample(guide=guide, text=sentence, label=label, offsets=offsets))
        return examples

    def _read_data(self, file_name, task='train', use_bio=True, sents=None):
        """
            data file should be formatted as standard BIO or BIES
            loading train and dev using task='train' where label will be load as well
            loading test using task='test' where all labels will be 'O'
            
            Input:
            
            file_name (Path): full file path
            task (str): 'train' or 'test'
            use_bio (bool): True if file_name is a bio file, False if file_name is normalized text
        """
        assert task in {'train', 'test'}, 'task shoud be either train or test but got {}'.format(task)
        
        def word_to_samples(words, unique_labels, splited=False):
            nsent, offsets, labels = [], [], []
            for j, word in enumerate(words):
                word_info = word.split(" ") if not splited else word
                if len(word_info) < 2:
                    warnings.warn("""
                    The word at [line: {} pos: {}] has no position or label information.
                    This word will cause potential runtime error.
                    Check if you have extra empty line between two sentences in the BIO files.
                    To avoid runtime error, we skip this word.
                    If skipping is not correct, please check the data_utils.py at line 130.""".format(i+1, j+1),
                                RuntimeWarning)
                    continue
                nsent.append(word_info[0])
                if self.has_offset_info:
                    # In the BIO data, we require to record two sets of offsets (4 numbers) after token text.
                    # e.g., Record 0 4 0 4 O
                    # In the example, the first two numbers are original offsets,
                    # the second two numbers are new offsets after pre-processing.
                    # If your do not conduct pre-processing and only use the original offsets,
                    # you can just set new offsets after pre-processing as original offsets to avoid index error.
                    warnings.warn("""
                    In the BIO data, we require to record two sets of offsets (4 numbers) after token text.
                    e.g., Record 0 4 0 4 O
                    In the example, the first two numbers are original offsets,
                    the second two numbers are new offsets after pre-processing.
                    If your do not conduct pre-processing and only use the original offsets,
                    you can just set new offsets after pre-processing as original offsets to avoid index error.""")
                    offsets.append((word_info[1], word_info[2], word_info[3], word_info[4]))
                if task == "train":
                    unique_labels.add(word_info[-1])
                    labels.append(word_info[-1])
                else:
                    labels.append("O")
            return (nsent, offsets, labels)
        
        if not use_bio:
            if sents is None:
                nsents = []
                txt, sents = pre_processing(file_name, deid_pattern="\[\*\*|\*\*\]", sent_tokenizer=self.sent_tokenizer)
                sents, _ = generate_BIO(sents, [], no_overlap=False)
            nsents, unique_labels = [], set()
            for sent in sents:
                words = [list(map(str, sum(list(map(lambda x: list(x) if (isinstance(x, list) or isinstance(x, tuple)) else [x], word)),[]))) for word in sent]
                out_tup = word_to_samples(words, unique_labels, splited=True)
                nsents.append(out_tup)
                
        else:
            with open(file_name, "r") as f:
                txt = f.read().strip()

                if len(txt) < 1:
                    warnings.warn(f"{file_name} is an empty file.")
                    return [], set()

                nsents, unique_labels = [], set()
                sents = txt.split("\n\n")

                for i, sent in enumerate(sents):
                    words = sent.split("\n")
                    out_tup = word_to_samples(words, unique_labels)
                    nsents.append(out_tup)

        return nsents, unique_labels


def __seq2fea(new_tokens, new_labels, guards, tokenizer, max_seq_length, label2idx):
    if isinstance(tokenizer, BertTokenizer) or \
            isinstance(tokenizer, DistilBertTokenizer) or \
            isinstance(tokenizer, ElectraTokenizer) or \
            isinstance(tokenizer, DebertaV2Tokenizer) or \
            isinstance(tokenizer, DebertaTokenizer):
        s_tk, e_tk, pad_tk = '[CLS]', '[SEP]', '[PAD]'
    elif isinstance(tokenizer, RobertaTokenizer) or \
            isinstance(tokenizer, BartTokenizer) or \
            isinstance(tokenizer, LongformerTokenizer):
        s_tk, e_tk, pad_tk = '<s>', '</s>', '<pad>'
    elif isinstance(tokenizer, XLNetTokenizer):
        s_tk, e_tk, pad_tk = '<cls>', '<sep>', '<pad>'
    elif isinstance(tokenizer, AlbertTokenizer):
        s_tk, e_tk, pad_tk = '[CLS]', '[SEP]', '<pad>'
    else:
        raise RuntimeError("The current package does not support tokenizer: {}.".format(type(tokenizer)))

    # add special tokens - bert robert insert at the begining, xlnet append at the end
    if isinstance(tokenizer, XLNetTokenizer):
        # single sequence: ``<s>... pad pad X <\s> sep cls``
        new_tokens.insert(0, '<s>')
        new_labels.insert(0, 'O')
        guards.insert(0, 0)
        new_tokens.append('</s>')
        new_labels.append('O')
        guards.append(0)
        new_tokens.append(e_tk)
        new_labels.append('O')
        guards.append(0)
        new_tokens.append(s_tk)
        new_labels.append('O')
        guards.append(0)
    else:
        # single sequence: ``cls X sep pad ...``
        new_tokens.insert(0, s_tk)
        new_labels.insert(0, 'CLS')
        guards.insert(0, 0)
        new_tokens.append(e_tk)
        # new_labels.append('SEP')
        new_labels.append('O')
        guards.append(0)

    # convert tokens to token ids; labels to label ids
    masks = [1] * len(new_tokens)
    cur_len = len(new_tokens)

    # padding - note for xlnet padding will be pre-padding instead of post-padding for all others
    # this is the implementation from Huggingface design, we will follow their pattern.
    pad_len = 0
    if isinstance(tokenizer, XLNetTokenizer):
        while cur_len < max_seq_length:
            new_tokens.insert(0, pad_tk)
            new_labels.insert(0, 'PAD')
            masks.insert(0, 0)
            guards.insert(0, 0)  # this wil lead to a problem in prediction
            cur_len += 1
            pad_len += 1
    else:
        while cur_len < max_seq_length:
            new_tokens.append(pad_tk)
            new_labels.append('PAD')
            masks.append(0)
            guards.append(0)
            cur_len += 1

    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    label_ids = list(map(lambda x: label2idx[x] if x in label2idx else label2idx['O'], new_labels))

    if isinstance(tokenizer, XLNetTokenizer):
        # for single sequence, we do not need to assign PAD token_type_ids to 3
        segment_ids = [3] * pad_len + [0] * (max_seq_length - (pad_len + 1)) + [2]
    else:
        segment_ids = [0] * max_seq_length

    assert len(new_tokens) + len(new_token_ids) + len(label_ids) + len(segment_ids) + len(masks) == 5 * max_seq_length, \
        f'''
            check feature generation; expect all data len = {max_seq_length} but 
            tokens: {len(new_tokens)},
            token_ids: {len(new_token_ids)}
            labels: {len(label_ids)},
            segments: {len(segment_ids)},
            masks: {len(masks)}.
        '''

    single_feature = InputFeature(input_tokens=new_tokens,
                                  input_ids=new_token_ids,
                                  attention_mask=masks,
                                  segment_ids=segment_ids,
                                  label=label_ids,
                                  guards=guards)
    return single_feature


def find_idx(xs, i):
    prev = xs[i]
    for j, x in enumerate(reversed(xs[:i])):
        if prev != x:
            return i - j
        prev = x
    return i


def _transformer_convert_data_to_features_helper(args, raw_tokens, labels, guide, label2idx, tokenizer, max_seq_length):
    """
        process a single example to feature (tokenization sentence and rematch labels)
        e.g., input:  (I love coding, [B-PER, O, B-ACT])
              output: ([<s>, I, love, Ġcod, ing, </s>, <PAD>], [SPLITTER, B-PER, O, B-ACT, X, SPLITTER, PAD])

        For training and dev, we will cut the sequence to max_seq_len - 2 to fit the model.
        For testing, since we can afford to loss any information. In this case, we will add the control token '[NEXT]' to indicate the current sentence should follow the next one as a whole seq.
        We really recommend to deal with long sentence in corpus instead of rely on sentence split in this function
    """
    new_tokens, new_labels, guards = [], [], []
    for i, (raw_token, label) in enumerate(zip(raw_tokens, labels)):
        new_token = tokenizer.tokenize(raw_token)
        new_tokens.extend(new_token)
        for k, _ in enumerate(new_token):
            if k == 0:
                new_labels.append(label)
            else:
                if isinstance(tokenizer, XLNetTokenizer):
                    new_labels.append(label)
                else:
                    new_labels.append('X')
            guards.append(i+1)

    processed_feature = []
    tlen = len(new_tokens)
    if isinstance(tokenizer, XLNetTokenizer):
        shifts = 4
    else:
        shifts = 2
    # shifts = 2
    if tlen > max_seq_length - shifts:
        args.logger.warning(f'''
Sentence after tokenization is too long (expect less than {max_seq_length - 2} but got {tlen}).
You have to increase max_seq_length to make sure every sentence after tokenization is not longer than the max_seq_length.
Or you can further split you long sentences into shorter ones. We will defaultly segment the sentence to two or more seqs with length as (~max_seq_len * n + rest).
The extra long sentence: 
{' '.join(raw_tokens)}
''')
        while tlen > max_seq_length - shifts:
            cutoff = find_idx(guards, max_seq_length - shifts - 1)
            tlen -= cutoff
            _new_tokens = new_tokens[:cutoff] + [NEXT_TOKEN]
            _new_labels = new_labels[:cutoff] + ['O']
            _guards = guards[:cutoff] + [NEXT_GUARD]
            single_feature = __seq2fea(_new_tokens, _new_labels, _guards, tokenizer, max_seq_length, label2idx)
            processed_feature.append(single_feature)
            new_tokens, new_labels, guards = new_tokens[cutoff:], new_labels[cutoff:], guards[cutoff:]
        single_feature = __seq2fea(new_tokens, new_labels, guards, tokenizer, max_seq_length, label2idx)
        processed_feature.append(single_feature)
    else:
        single_feature = __seq2fea(new_tokens, new_labels, guards, tokenizer, max_seq_length, label2idx)
        processed_feature.append(single_feature)

    return processed_feature


def transformer_convert_data_to_features(args, input_examples, label2idx, tokenizer=None, max_seq_len=128):
    features = []
    for idx, example in enumerate(input_examples):
        raw_tokens, labels, guide = example.text, example.label, example.guide.split("-")[0]
        feature = _transformer_convert_data_to_features_helper(args, raw_tokens, labels, guide, label2idx, tokenizer, max_seq_len)
        features.extend(feature)
        fea = feature[0]
        if idx < 3:
            args.logger.info("""
************** Example ************
input tokens: {}
input ids: {}
label: {}
mask: {}
segment ids: {}
guards: {}
***********************************
            """.format(fea.input_tokens, fea.input_ids, fea.label, fea.attention_mask, fea.token_type_ids, fea.guards))

    return features


def convert_features_to_tensors(features):
    tensor_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    tensor_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    tensor_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    tensor_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    tensor_guards = torch.tensor([f.guards for f in features], dtype=torch.long)

    return TensorDataset(tensor_input_ids, tensor_attention_masks, tensor_token_type_ids, tensor_label_ids, tensor_guards)


def ner_data_loader(dataset, batch_size=2, task='train', auto=False):
    """
    task has two levels:
    train for training using RandomSampler
    test for evaluation and prediction using SequentialSampler

    if set auto to True we will default call convert_features_to_tensors,
    so features can be directly passed into the function
    """
    if auto:
        dataset = convert_features_to_tensors(dataset)

    if len(dataset) < 1:
        return DataLoader(dataset)

    if task == 'train':
        sampler = RandomSampler(dataset)
    elif task == 'test':
        sampler = SequentialSampler(dataset)
    else:
        raise ValueError('task argument only support train or test but get {}'.format(task))

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)
    return data_loader


def batch_to_model_inputs(batch, model_type='bert'):
    if model_type in {"bert", "megatron", "albert", "distilbert", "xlnet", "electra", 'deberta', 'deberta-v2'}:
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'label_ids': batch[3]
        }
    elif model_type in {"roberta", "bart", "longformer"}:
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'label_ids': batch[3]
        }
    else:
        raise RuntimeError(f"{model_type} is not a valid model type")

    return inputs
