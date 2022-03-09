# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

from pathlib import Path
import numpy as np
from functools import partial
from transformers import BertTokenizer
import warnings
import tqdm
import multiprocessing
import traceback
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from common_utils.common_io import read_from_file, pkl_load, pkl_dump, json_load


class InputFeature(object):
    def __init__(self, input_tokens, input_ids, attention_masks, token_type_ids, token_sub_indexing, labels, masks):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.token_sub_indexing = token_sub_indexing
        self.labels = labels
        self.masks = masks

    def __repr__(self):
        return str(self.__dict__)


def batch_to_model_inputs(batch, device=None):
    if device:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'token_type_ids': batch[2].to(device),
                  'labels': batch[3].to(device),
                  'masks': batch[4].to(device)}
    else:
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3],
                  'masks': batch[4]}
    return inputs


####################### multiprocessing feature to tensor process #######################
# def _f2t(features, return_dict, dtype):
#     try:
#         if dtype == "input":
#             return_dict[dtype] = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#         elif dtype == "att_mask":
#             return_dict[dtype] = torch.tensor([f.attention_masks for f in features], dtype=torch.long)
#         elif dtype == "segment":
#             return_dict[dtype] = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
#         elif dtype == "labels":
#             return_dict[dtype] = torch.tensor([f.labels for f in features], dtype=torch.long)
#         elif dtype == "masks":
#             return_dict[dtype] = torch.tensor([f.masks for f in features], dtype=torch.long)
#     except Exception as ex:
#         traceback.print_exc()


# def convert_features_to_tensors(features):
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()

#     p1 = multiprocessing.Process(target=_f2t, args=(features, return_dict, "input"))
#     p2 = multiprocessing.Process(target=_f2t, args=(features, return_dict, "att_mask"))
#     p3 = multiprocessing.Process(target=_f2t, args=(features, return_dict, "segment"))
#     p4 = multiprocessing.Process(target=_f2t, args=(features, return_dict, "labels"))
#     p5 = multiprocessing.Process(target=_f2t, args=(features, return_dict, "masks"))
#     jobs = [p1, p2, p3, p4, p5]

#     for proc in jobs:
#         proc.start()

#     for proc in jobs:
#         proc.join()

#     tensor_input_ids = return_dict["input"]
#     tensor_attention_masks = return_dict["att_mask"]
#     tensor_token_type_ids = return_dict["segment"]
#     tensor_labels = return_dict["labels"]
#     tensor_masks = return_dict["masks"]

#     return TensorDataset(tensor_input_ids,
#                          tensor_attention_masks,
#                          tensor_token_type_ids,
#                          tensor_labels,
#                          tensor_masks)
####################### multiprocessing feature to tensor process #######################


def convert_features_to_tensors(features):
    print("convert features to tensors ...")

    tensor_input_ids = torch.tensor(
        np.array([f.input_ids for f in features]), dtype=torch.long)
    tensor_attention_masks = torch.tensor(
        np.array([f.attention_masks for f in features]), dtype=torch.long)
    tensor_token_type_ids = torch.tensor(
        np.array([f.token_type_ids for f in features]), dtype=torch.long)
    tensor_sub_index_ids = torch.tensor(
        np.array([f.token_sub_indexing for f in features]), dtype=torch.long)
    tensor_labels = torch.tensor(
        np.array([f.labels for f in features]), dtype=torch.long)
    tensor_masks = torch.tensor(
        np.array([f.masks for f in features]), dtype=torch.long)

    print("convert features to tensors done.")

    return TensorDataset(tensor_input_ids,
                         tensor_attention_masks,
                         tensor_token_type_ids,
                         tensor_labels,
                         tensor_masks,
                         tensor_sub_index_ids)


class TransformerNerBiaffineDataProcessor(object):
    def __init__(self, data_dir=None, logger=None, tokenizer=None, max_seq_len=512, cache=True, tokenizer_type=None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache = cache
        self.tokenizer_type = tokenizer_type
        # label2idx: we do not need to distiguish all un-entity tokens, set all as 0
        self.label2idx = {'O': 0, 'X': 0, 'PAD': 0, 'CLS': 0, 'SEP': 0}

    def set_cache(self, cache):
        self.cache = cache

    def set_logger(self, logger):
        self.logger = logger

    def set_data_dir(self, data_dir):
        self.data_dir = Path(data_dir)

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_tokenizer_type(self, tokenizer_type):
        self.tokenizer_type = tokenizer_type

    def set_label2idx(self, label2idx):
        self.label2idx = label2idx

    def get_train_examples(self):
        input_file_name = self.data_dir / "train.json"
        data, _unique_labels = self._read_data(input_file_name, task='train')
        return data, _unique_labels

    def get_dev_examples(self):
        input_file_name = self.data_dir / "dev.json"
        data, _ = self._read_data(input_file_name, task='test')
        return data, None

    def get_test_examples(self):
        input_file_name = self.data_dir / "test.json"
        data, _ = self._read_data(input_file_name, task='test')
        return data, None

    def _get_unique_labels(self):
        return self.get_train_examples()[-1]

    def get_labels(self, labels_file=None):
        # you can create a labels file to define all the possible entity types
        # if no labels file provided, we use training data to extract all unique labels
        if labels_file and Path(labels_file).exists():
            # we assume each line is a type
            labels = read_from_file(labels_file).strip().split("\n")
        else:
            labels = self._get_unique_labels()

        # new lable id start from 1 since all other default labels are 0
        label_id = 1
        for label in labels:
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                label_id += 1

        # we return label2idx map and all labels
        return self.label2idx, list(self.label2idx.keys())

    def _read_data(self, file_name, task='train'):
        assert task in {'train', 'test'}, 'task should be either train or test but got {}'.format(task)
        data = json_load(file_name)

        if len(data) == 0:
            self.logger.warning(f"{file_name} is empty")
            return [], set()

        unique_en_types = set()
        if task == "train":
            # only get unique entity types when it is training data
            for each in data:
                # {tokens: [], entities: [['Vicodin', 'treatment', (1, 1)]]}
                s = set([en[1] for en in each['entities']])
                unique_en_types.update(s)

        return data, unique_en_types

    def _tokens2ids(self, tokens, token_sub_indexing):
        if self.tokenizer_type in {"bert", "distilbert", "electra", "deberta", "deberta-v2", "megatron"}:
            s_tk, e_tk, pad_tk = '[CLS]', '[SEP]', '[PAD]'
        elif self.tokenizer_type in {'roberta', 'longformer', 'bart'}:
            s_tk, e_tk, pad_tk = '<s>', '</s>', '<pad>'
        elif self.tokenizer_type == "xlnet":
            raise NotImplementedError("We do not support XLNet for Biaffine NER.")
        elif self.tokenizer_type == "albert":
            s_tk, e_tk, pad_tk = '[CLS]', '[SEP]', '<pad>'
        else:
            raise RuntimeError("The current package does not support tokenizer: {}.".format(self.tokenizer_type))

        # single sequence: ``cls X X X sep pad ...``
        tokens.insert(0, s_tk)
        token_sub_indexing.insert(0, 0)
        tokens.append(e_tk)
        token_sub_indexing.append(0)

        attention_masks = [1] * len(tokens)
        cur_len = len(tokens)

        while cur_len < self.max_seq_len:
            tokens.append(pad_tk)
            token_sub_indexing.append(0)
            attention_masks.append(0)
            cur_len += 1

        new_token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * self.max_seq_len

        assert len(new_token_ids) + len(token_type_ids) + len(attention_masks) == self.max_seq_len * 3, \
            "check feature generation, ids are not the same size as max_seq_len, maybe max_seq_len is not enough."

        return new_token_ids, attention_masks, token_type_ids, token_sub_indexing

    def _update_labels(self, entities, mapping):
        new_entities = []
        self.logger.debug("remapping the indexes of entities after tokenization...")
        for en in entities:
            text, ty, s_e = en[:3]
            s, e = s_e
            # we need to add 1 because [CLS] insertion
            new_s = mapping[s][0] + 1
            new_e = mapping[e][-1] + 1
            new_entities.append([ty, (new_s, new_e)])
        return new_entities

    def _create_labels_and_masks(self, entities, attention_masks):
        # labels dim = max_seq_len * max_seq_len
        labels = np.zeros((self.max_seq_len, self.max_seq_len), dtype=int)
        for en in entities:
            label_id = self.label2idx[en[0]]
            s, e = en[1]
            labels[s, e] = label_id

        # masks dim = max_seq_len * max_seq_len
        masks = [attention_masks for _ in range(sum(attention_masks))]
        line_zeros = [0] * self.max_seq_len
        masks.extend([line_zeros for _ in range(len(masks), self.max_seq_len)])
        masks = np.array(masks)

        return labels, masks

    def data2feature_parallel(self, examples, task="test"):
        from concurrent.futures import ProcessPoolExecutor
        features = []
        batch_examples = np.array_split(examples, 8)
        with ProcessPoolExecutor(max_workers=8) as pool:
            for batch_features in pool.map(partial(self.data2feature, task=task), batch_examples):
                features.extend(batch_features)

        return features

    def data2feature(self, examples, task="test"):
        features = []
        for example in tqdm.tqdm(examples, desc="examples2features"):
            tok_remap = dict()
            tokens = example["tokens"]
            entities = example["entities"]
            new_tokens = []
            token_sub_indexing = []
            for idx, tok in enumerate(tokens):
                new_toks = self.tokenizer.tokenize(tok)
                tok_remap[idx] = (len(new_tokens), len(new_tokens) + len(new_toks) - 1)
                new_tokens.extend(new_toks)
                token_sub_indexing.extend([(idx + 1)] * len(new_toks))

            if len(new_tokens) > (self.max_seq_len - 4):
                warnings.warn(
                    f"example: {example}\nthe tokens are too many and will cause truncate which leads to error\n \
                    check preprocessing to make the sentences shorter.")

            new_token_ids, attention_masks, token_type_ids, token_sub_indexing = self._tokens2ids(
                new_tokens, token_sub_indexing)

            # Do we need to create labels and masks here? quite MEM intense; also take too much space to save cache
            # TODO: consider generating labels/masks using data collate_fn on fly
            # create 2D labels and masks based on token remapping and attention_masks
            new_entities = self._update_labels(entities, tok_remap)
            labels, masks = self._create_labels_and_masks(new_entities, attention_masks)

            feature = InputFeature(input_tokens=" ".join(new_tokens),
                                   input_ids=new_token_ids,
                                   attention_masks=attention_masks,
                                   token_type_ids=token_type_ids,
                                   token_sub_indexing=token_sub_indexing,
                                   labels=labels,
                                   masks=masks
                                   )
            features.append(feature)

        return features

    def _data_loader(self, task="train", batch_size=4, to_tensor=True):
        # process data to features
        if task == "train":
            data, _ = self.get_train_examples()
        elif task == "dev":
            data, _ = self.get_dev_examples()
        else:
            data, _ = self.get_test_examples()

        # cache if possible
        tokenizer_name = self.tokenizer.name_or_path.split("/")[-1]
        fn = self.data_dir / f"cache_{tokenizer_name}_{task}_{self.max_seq_len}.pkl"
        if fn.exists():
            self.logger.info(f"load {task} data from {fn}")
            dataset = pkl_load(fn)
        else:
            # dataset = self.data2feature(data, task)
            # #use parallel with 8 cores
            dataset = self.data2feature_parallel(data, task)
        if self.cache and not fn.exists():
            self.logger.info(f"set cache to True so it will save {task} data at {fn}")
            pkl_dump(dataset, fn)

        # task can be train, dev, test
        if to_tensor:
            dataset = convert_features_to_tensors(dataset)

        if task == "train":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        # we hard code the num_workers as 4; can be increased to 8 (4 is ok give 90% GPU utilization on avg)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True, num_workers=4)

        return data_loader

    def get_train_data_loader(self, batch_size=4):
        return self._data_loader(task="train", batch_size=batch_size)

    def get_test_data_loader(self, batch_size=4, to_tensor=True):
        return self._data_loader(task="test", batch_size=batch_size)

    def get_dev_data_loader(self, batch_size=4, to_tensor=True):
        return self._data_loader(task="dev", batch_size=batch_size)


if __name__ == "__main__":
    import logging

    dp = TransformerNerBiaffineDataProcessor(data_dir="./data/i2b22010/")
    m = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(m)
    dp.set_tokenizer(tokenizer=tokenizer)
    dp.set_logger(logger=logging.getLogger())
    dp.set_max_seq_len(512)

    print(dp.get_labels())

    for i, each in enumerate(dp.get_dev_data_loader(batch_size=3)):
        if i < 5:
            print(each[0].shape, each[-2].shape, each[-1].shape)
