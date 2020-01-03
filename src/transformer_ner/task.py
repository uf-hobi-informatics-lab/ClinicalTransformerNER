#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
We will not use multi-GPUs training for NER (no n_gpu arg anymore)
If we will add multi-GPUs training, using ditributed method (local_rank)

We will support cache and load cache data in future but not implemented here since clinical NER dataset is relatively small)

Though new model will be trained and saved separately, we use the tokenizer and config from base model without modification. Therefore, when run prediction separately, we need to set --config_name
and --tokenizer_name to base model name (e.g., bert-base-uncased). The base model name can be found in model directory. We will automatically set up this in later version.
"""

from common_utils.bio_prf_eval import BioEval
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertTokenizer,
                          XLNetConfig, XLNetTokenizer,
                          RobertaConfig, RobertaTokenizer,
                          AlbertConfig, AlbertTokenizer,
                          DistilBertConfig, DistilBertTokenizer)
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import random
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm, trange
from common_utils.common_io import json_load, output_bio, json_dump

from transformer_ner.model import BertNerModel, RobertaNerModel, XLNetNerModel, AlbertNerModel, DistilBertNerModel, BertLikeNerModel
from transformer_ner.data_utils import (TransformerNerDataProcessor, transformer_convert_data_to_features,
                                        ner_data_loader, batch_to_model_inputs,
                                        convert_features_to_tensors, NEXT_GUARD)


MODEL_CLASSES = {
    'bert': (BertConfig, BertNerModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetNerModel, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaNerModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertNerModel, AlbertTokenizer),
    'distilbert': (DistilBertConfig, DistilBertNerModel, DistilBertTokenizer)
}


def load_model(model_dir):
    model_dir = Path(model_dir)
    all_model_files = model_dir.glob("*.bin")
    all_model_files = list(filter(lambda x: 'checkpoint_' in x.as_posix(), all_model_files))
    sorted_file = sorted(all_model_files, key=lambda x: int(x.stem.split("_")[-1]))
    return torch.load(sorted_file[-1], map_location=torch.device('cpu'))


def save_only_transformer_core(args, model):
    if args.model_type == "bert":
        model_core = model.bert
    elif args.model_type == "roberta":
        model_core = model.roberta
    elif args.model_type == "xlnet":
        model_core = model.transformer
    elif args.model_type == "distilbert":
        model_core = model.distilbert
    elif args.model_type == "albert":
        model_core = model.albert
    else:
        args.logger.warning("{} is current not supported for saving model core; we will skip saving to prevent error.".format(args.model_type))
        return
    model_core.save_pretrained(args.new_model_dir)


def save_model(model, model_dir, index, latest=3):
    model_to_save = model.module if hasattr(model, 'module') else model
    new_model_file = model_dir / "checkpoint_{}.bin".format(index)
    torch.save(model_to_save, new_model_file)
    # only keep 'latest' # of checkpoints
    all_model_files = list(model_dir.glob("*.bin"))
    if len(all_model_files) > latest:
        # sorted_files = sorted(all_model_files, key=lambda x: os.path.getmtime(x))  # sorted by the last modified time
        sorted_file = sorted(all_model_files, key=lambda x: int(x.stem.split("_")[-1]))  # sorted by the checkpoint step
        file_to_remove = sorted_file[0]  # remove earliest checkpoints
        file_to_remove.unlink()


def check_partial_token(token_as_id, tokenizer):
    token = tokenizer.convert_ids_to_tokens(int(token_as_id))
    flag = False
    if (isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, DistilBertTokenizer)) and token.startswith('##'):
        flag = True
    elif isinstance(tokenizer, RobertaTokenizer) and not token.startswith('Ġ'):
        flag = True
    elif isinstance(tokenizer, XLNetTokenizer) and not token.startswith('▁'):
        flag = True
    elif isinstance(tokenizer, AlbertTokenizer) and not token.startswith("▁"):
        flag = True
    return flag


def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args, model, train_features, dev_features):
    """NER model training on train dataset; select model based on performance on dev dataset"""
    # create data loader
    data_loader = ner_data_loader(train_features, batch_size=args.train_batch_size, task='train', auto=True)
    # total training step counts
    t_total = len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # parameters for optimization
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # using fp16 for training rely on Nvidia apex package
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # training linear warm-up setup
    scheduler = None
    if args.do_warmup:
        warmup_steps = np.dtype('int64').type(args.warmup_ratio * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    args.logger.info("***** Running training *****")
    args.logger.info("  Num data points = {}".format(len(data_loader)))
    args.logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.train_batch_size))
    args.logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    args.logger.info("  Total optimization steps = {}".format(t_total))
    args.logger.info("  Training steps (number of steps between two evaluation on dev) = {}".format(args.train_steps))
    args.logger.info("******************************")

    # create directory to save model
    new_model_dir = Path(args.new_model_dir)
    new_model_dir.mkdir(parents=True, exist_ok=True)
    # save label2idx json in new model directory
    json_dump(args.label2idx, new_model_dir / "label2idx.json")

    # save base model name to a base_model_name.txt
    with open(new_model_dir/"base_model_name.txt", "w") as f:
        f.write('model_type: {}\nbase_model: {}\nconfig: {}\ntokenizer: {}'.format(args.model_type, args.pretrained_model, args.config_name, args.tokenizer_name))

    global_step = 0
    tr_loss = .0
    best_score, epcoh_best_score = .0, .0
    early_stop_flag = 0

    model.zero_grad()
    epoch_iter = trange(int(args.num_train_epochs), desc="Epoch", disable=False if args.progress_bar else True)
    for epoch in epoch_iter:
        batch_iter = tqdm(iterable=data_loader, desc='Batch', disable=False if args.progress_bar else True)
        for step, batch in enumerate(batch_iter):
            model.train()
            batch = tuple(b.to(args.device) for b in batch)
            train_inputs = batch_to_model_inputs(batch)
            _, _, loss = model(**train_inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.do_warmup:
                    scheduler.step()
                model.zero_grad()
                global_step += 1

            # using training step
            if args.train_steps > 0 and (global_step + 1) % args.train_steps == 0:
                best_score, eval_loss = evaluate(args, model, new_model_dir, dev_features, epoch, global_step, best_score)
                args.logger.info("""
                Global step: {}; 
                Epoch: {}; 
                average_train_loss: {:.4f}; 
                eval_loss: {:.4f}; 
                current best score: {:.4f}""".format(global_step,epoch,round(tr_loss / global_step, 4), eval_loss,best_score))

        # default model select method using strict F1-score with beta=1; evaluate model after each epoch on dev
        if args.train_steps <= 0:
            best_score, eval_loss = evaluate(args, model, new_model_dir, dev_features, epoch, global_step, best_score)
            args.logger.info("""
                Global step: {}; 
                Epoch: {}; 
                average_train_loss: {:.4f}; 
                eval_loss: {:.4f}; 
                current best score: {:.4f}""".format(global_step, epoch, round(tr_loss / global_step, 4), eval_loss, best_score))

        # early stop check
        if epcoh_best_score < best_score:
            epcoh_best_score = best_score
            early_stop_flag = 0
        else:
            early_stop_flag += 1

        if 0 < args.early_stop <= early_stop_flag:
            args.logger.warn('Early stop activated; performance not improve anymore.')
            break


def _eval(args, model, features):
    """common evaluate test data with pre-trained model shared by eval and predict"""
    data_loader = ner_data_loader(features, batch_size=args.eval_batch_size, task='test', auto=True)
    eval_size = len(data_loader)
    args.logger.info("***** Running evaluation on {} number of test data *****".format(eval_size))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.eval_batch_size))
    args.logger.info("******************************")

    # prepare processing results for each batch
    y_trues, y_preds = [], []
    y_pred, y_true = [], []
    prev_gd = 0

    # prediction
    model.eval()
    eval_loss = .0
    for batch in tqdm(data_loader, desc='evaluation', disable=False if args.progress_bar else True):
        original_tkid = batch[0].numpy()
        original_mask = batch[1].numpy()
        original_labels = batch[3].numpy()
        guards = batch[4].numpy()

        batch = tuple(b.to(args.device) for b in batch)
        eval_inputs = batch_to_model_inputs(batch)

        with torch.no_grad():
            raw_logits, _, loss = model(**eval_inputs)
            # get softmax output of the raw logits (keep dimensions)
            raw_logits = torch.argmax(F.log_softmax(raw_logits, dim=2), dim=2)
            raw_logits = raw_logits.detach().cpu().numpy()
            # update evaluate loss
            eval_loss += loss.item()

        assert guards.shape == original_tkid.shape == original_mask.shape == original_labels.shape == raw_logits.shape,  \
            """
                expect same dimension for all the inputs and outputs but get
                input_tokens: {}
                mask: {}
                label: {}
                output logits:{}
            """.format(original_tkid.shape, original_mask.shape, original_labels.shape, raw_logits.shape)

        # tk=token, mk=mask, lb=label, lgt=logits
        for mks, lbs, lgts, gds in zip(original_mask, original_labels, raw_logits, guards):
            connect_sent_flag = False
            for mk, lb, lgt, gd in zip(mks, lbs, lgts, gds):
                if mk == 0:  # after hit first mask, we can stop for the current sentence since all rest will be pad
                    break
                if gd == 0 or prev_gd == gd:
                    continue
                if gd == NEXT_GUARD:
                    connect_sent_flag = True
                    break
                if prev_gd != gd:
                    y_true.append(args.idx2label[lb])
                    y_pred.append(args.idx2label[lgt])
                    prev_gd = gd
            if connect_sent_flag:
                continue
            y_trues.append(y_true)
            y_preds.append(y_pred)
            y_pred, y_true = [], []
            prev_gd = 0

    return y_trues, y_preds, round(eval_loss/eval_size, 4)


def evaluate(args, model, new_model_dir, features, epoch, global_step, best_score):
    """evaluationn dev dataset and return acc, pre, rec, f1 score for model development"""
    y_true, y_pred, eval_loss = _eval(args, model, features)
    args.eval_tool.eval_mem(y_true, y_pred)
    # # # debug
    # args.logger.debug(args.eval_tool.get_counts())
    # args.logger.debug(args.eval_tool.get_performance())
    eval_metrix = args.eval_tool.get_performance()
    score_lvl, score_method, _ = args.model_selection_scoring.split("-")
    cur_score = eval_metrix['overall'][score_lvl][score_method]
    args.eval_tool.reset()

    if best_score < cur_score:
        args.logger.info('''
        Global step: {}; 
        Epoch: {}; 
        previous best score: {:.4f}; 
        new best score: {:.4f}; 
        full evaluation metrix: {}
        '''.format(global_step, epoch, best_score, cur_score, eval_metrix))
        best_score = cur_score
        save_model(model, new_model_dir, global_step, latest=args.max_num_checkpoints)

    return best_score, eval_loss


def predict(args, model, features):
    """evaluation test dataset and output predicted results as BIO"""
    _, y_pred, _ = _eval(args, model, features)
    # fix prediction of system labels (X, SEP, CLS etc.)
    system_labels = {label for idx, label in args.idx2label.items() if idx < args.label2idx['O']}
    fixed_preds = []
    for each in y_pred:
        fixed_preds.append(list(map(lambda x: 'O' if x in system_labels else x, each)))
    return fixed_preds


def _output_bio(args, tests, preds):
    new_sents = []
    assert len(tests) == len(preds), "Expect {} sents but get {} sents in prediction".format(len(tests), len(preds))
    for example, predicted_labels in zip(tests, preds):
        tokens = example.text
        assert len(tokens) == len(predicted_labels), "Not same length sentence\nExpect: {} {}\nBut: {} {}".format(len(tokens), tokens, len(predicted_labels), predicted_labels)
        offsets = example.offsets
        if offsets:
            new_sent = [(tk, ofs[0], ofs[1], lb) for tk, ofs, lb in zip(tokens, offsets, predicted_labels)]
        else:
            new_sent = [(tk, lb) for tk, lb in zip(tokens, predicted_labels)]
        new_sents.append(new_sent)

    output_bio(new_sents, args.predict_output_file)
    args.logger.warning("""The output file is {}, we recommend to use suffix as .bio.txt or .txt. 
            You can directly use formatter coverter tool in this package.""".format(
        args.predict_output_file))


def set_up_eval_tool(args):
    bio_eval = BioEval()
    _, _, beta = args.model_selection_scoring.split("-")  # strict-f_score-1
    bio_eval.set_beta_for_f_score(beta=int(beta))
    bio_eval.set_logger(args.logger)
    eval_ignore_labels = [label for idx, label in args.idx2label.items() if idx < args.label2idx['O']]
    bio_eval.add_labels_not_for_eval(*eval_ignore_labels)
    return bio_eval


def run_task(args):
    set_seed(args.seed)

    if os.path.exists(args.new_model_dir) and os.listdir(args.new_model_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError('new model directory: {} exists. Use --overwrite_output_dir to overwrite the previous model or create another directory for the new model'.format(args.new_model_dir))

    # init data processor
    ner_data_processor = TransformerNerDataProcessor()
    ner_data_processor.set_data_dir(args.data_dir)
    ner_data_processor.set_logger(args.logger)
    if args.data_has_offset_information:
        ner_data_processor.offset_info_available()

    if args.do_train:
        labels, label2idx = ner_data_processor.get_labels(default=args.model_type)
    else:
        label2idx = json_load(os.path.join(args.new_model_dir, "label2idx.json"))

    num_labels = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    args.label2idx = label2idx
    args.idx2label = idx2label

    # get config, model and tokenizer
    model_config, model_model, model_tokenizer = MODEL_CLASSES[args.model_type]
    args.logger.info("Training/evaluation parameters: {}".format({k: v for k, v in vars(args).items()}))

    # training
    if args.do_train:
        config = model_config.from_pretrained(args.config_name, num_labels=num_labels)
        model = model_model.from_pretrained(args.pretrained_model, from_tf=bool('.ckpt' in args.pretrained_model), config=config)
        tokenizer = model_tokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

        # add an control token for combine sentence if it is too long to fit max_seq_len
        args.tokenizer = tokenizer
        model.to(args.device)

        train_examples = ner_data_processor.get_train_examples()
        train_features = transformer_convert_data_to_features(args,
                                                              input_examples=train_examples,
                                                              label2idx=label2idx,
                                                              tokenizer=tokenizer,
                                                              max_seq_len=args.max_seq_length)

        dev_examples = ner_data_processor.get_dev_examples()
        dev_features = transformer_convert_data_to_features(args,
                                                            input_examples=dev_examples,
                                                            label2idx=label2idx,
                                                            tokenizer=tokenizer,
                                                            max_seq_len=args.max_seq_length)

        # set up evaluation metrics
        args.eval_tool = set_up_eval_tool(args)
        # start training
        train(args, model, train_features, dev_features)
        # save model transformer core
        if args.save_model_core:
            save_only_transformer_core(args, model)
        # save config and tokenizer with new model
        tokenizer.save_pretrained(args.new_model_dir)
        config.save_pretrained(args.new_model_dir)

    # predict - test.txt file prediction (if you need predict many files, use 'run_transformer_batch_prediction')
    if args.do_predict:
        model = load_model(args.new_model_dir)
        tokenizer = model_tokenizer.from_pretrained(args.new_model_dir, do_lower_case=args.do_lower_case)
        args.tokenizer = tokenizer
        model.to(args.device)

        test_example = ner_data_processor.get_test_examples()
        test_features = transformer_convert_data_to_features(args,
                                                             input_examples=test_example,
                                                             label2idx=label2idx,
                                                             tokenizer=tokenizer,
                                                             max_seq_len=args.max_seq_length)

        predictions = predict(args, model, test_features)
        _output_bio(args, test_example, predictions)


def test():
    from pathlib import Path
    from transformer_ner.transfomer_log import TransformerNERLogger

    class Args:
        def __init__(self):
            self.model_type = 'bert'
            self.pretrained_model = 'bert-base-uncased'
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

    args = Args()
    run_task(args)

    from eval_scripts.old_bio_eval import main as eval_main
    eval_main(os.path.join(args.data_dir, "test.txt"), args.predict_output_file)


if __name__ == '__main__':
    test()
