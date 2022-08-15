#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
We will not use multi-GPUs training for NER (no n_gpu arg anymore)
If we will add multi-GPUs training, using ditributed method (local_rank)

We will support cache and load cache data in future but not implemented here since clinical NER dataset is relatively small)

Though new model will be trained and saved separately, we use the tokenizer and config from base model without modification. Therefore, when run prediction separately, we need to set --config_name
and --tokenizer_name to base model name (e.g., bert-base-uncased). The base model name can be found in model directory. We will automatically set up this in later version.
"""

import os
import random
import warnings
from pathlib import Path
import traceback

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import (AlbertConfig, AlbertTokenizer, BartConfig,
                          BartTokenizer, BertConfig, BertTokenizer,
                          DebertaConfig, DebertaTokenizer, DistilBertConfig,
                          DistilBertTokenizer, ElectraConfig, ElectraTokenizer,
                          LongformerConfig, LongformerTokenizer, RobertaConfig,
                          RobertaTokenizer, XLNetConfig, XLNetTokenizer,
                          DebertaV2Tokenizer, DebertaV2Config,
                          MegatronBertConfig)
# from transformers import get_linear_schedule_with_warmup

from common_utils.bio_prf_eval import BioEval
from common_utils.common_io import json_dump, json_load, output_bio
from transformer_ner.data_utils import (NEXT_GUARD, NEXT_TOKEN,
                                        TransformerNerDataProcessor,
                                        batch_to_model_inputs,
                                        convert_features_to_tensors,
                                        ner_data_loader,
                                        transformer_convert_data_to_features)
from transformer_ner.model import (AlbertNerModel, BartNerModel,
                                   BertNerModel, XLNetNerModel,
                                   DeBertaNerModel, DistilBertNerModel,
                                   ElectraNerModel, LongformerNerModel,
                                   RobertaNerModel, Transformer_CRF,
                                   DeBertaV2NerModel, MegatronNerModel)
from transformer_ner.model_utils import PGD, FGM, get_linear_schedule_with_warmup


MODEL_CLASSES = {
    'bert': (BertConfig, BertNerModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetNerModel, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaNerModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertNerModel, AlbertTokenizer),
    'distilbert': (DistilBertConfig, DistilBertNerModel, DistilBertTokenizer),
    'bart': (BartConfig, BartNerModel, BartTokenizer),
    'electra': (ElectraConfig, ElectraNerModel, ElectraTokenizer),
    'longformer': (LongformerConfig, LongformerNerModel, LongformerTokenizer),
    'deberta': (DebertaConfig, DeBertaNerModel, DebertaTokenizer),
    'deberta-v2': (DebertaV2Config, DeBertaV2NerModel, DebertaV2Tokenizer),
    'megatron': (MegatronBertConfig, MegatronNerModel, BertTokenizer)
}


ADVERSARIAL_TRAINER = {
    "pgd": PGD,
    "fgm": FGM
}


def load_model(args, new_model_dir=None):
    if not new_model_dir:
        model_dir = Path(args.new_model_dir)
    else:
        model_dir = Path(new_model_dir)
    all_model_files = model_dir.glob("*.bin")
    all_model_files = list(filter(lambda x: 'checkpoint_' in x.as_posix(), all_model_files))
    sorted_file = sorted(all_model_files, key=lambda x: int(x.stem.split("_")[-1]))
    # #load direct from model files
    args.logger.info("load checkpoint from {}".format(sorted_file[-1]))
    ckpt = torch.load(sorted_file[-1], map_location=torch.device('cpu'))
    # #load model as state_dict
    try:
        model = MODEL_CLASSES[args.model_type][1]
        model = model(config=args.config)
        model.load_state_dict(state_dict=ckpt)
    except AttributeError as Ex:
        args.logger.error(traceback.format_exc())
        args.logger.warning(
            """The model seems save using model.save instead of model.state_dict,
            attempt to directly using the loaded checkpoint as model.
            May raise error. If raise error, you need to check the model load whether is correct or not""")
        model = ckpt
    return model


def save_only_transformer_core(args, model):
    model_type = args.model_type
    if model_type == "bert":
        model_core = model.bert
    if model_type == "megatron":
        model_core = model.bert
    elif model_type == "roberta":
        model_core = model.roberta
    elif model_type == "xlnet":
        model_core = model.transformer
    elif model_type == "distilbert":
        model_core = model.distilbert
    elif model_type == "albert":
        model_core = model.albert
    elif model_type == "bart":
        model_core = model.bart
    elif model_type == "electra":
        model_core = model.electra
    elif model_type == "deberta":
        model_core = model.deberta
    elif model_type == "deberta-v2":
        model_core = model.deberta_v2
    elif model_type == "longformer":
        model_core = model.longformer
    else:
        args.logger.warning(
            "{} is current not supported for saving model core; we will skip saving to prevent error."
            .format(args.model_type))
        return
    model_core.save_pretrained(args.new_model_dir)


def save_model(args, model, model_dir, index, latest=3):
    new_model_file = model_dir / "checkpoint_{}.bin".format(index)
    args.tokenizer.save_pretrained(args.new_model_dir)
    args.config.save_pretrained(args.new_model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    # #direct save model
    # torch.save(model_to_save, new_model_file)
    # #save model as state dict
    args.logger.info("save checkpoint to {}".format(new_model_file))
    torch.save(model_to_save.state_dict(), new_model_file)
    # #only keep 'latest' # of checkpoints
    all_model_files = list(model_dir.glob("checkpoint_*.bin"))
    if len(all_model_files) > latest:
        # sorted_files = sorted(all_model_files, key=lambda x: os.path.getmtime(x))  # sorted by the last modified time
        sorted_file = sorted(all_model_files, key=lambda x: int(x.stem.split("_")[-1]))  # sorted by the checkpoint step
        file_to_remove = sorted_file[0]  # remove earliest checkpoints
        file_to_remove.unlink()


def check_partial_token(token_as_id, tokenizer):
    token = tokenizer.convert_ids_to_tokens(int(token_as_id))
    flag = False
    if (isinstance(tokenizer, BertTokenizer) or
        isinstance(tokenizer, DistilBertTokenizer) or
        isinstance(tokenizer, DebertaTokenizer) or
        isinstance(tokenizer, ElectraTokenizer)) \
            and token.startswith('##'):
        flag = True
    elif (isinstance(tokenizer, RobertaTokenizer) or
          isinstance(tokenizer, BartTokenizer) or
          isinstance(tokenizer, LongformerTokenizer)) \
            and not token.startswith('Ġ'):
        flag = True
    elif (isinstance(tokenizer, AlbertTokenizer) or
          isinstance(tokenizer, DebertaV2Tokenizer) or
          isinstance(tokenizer, XLNetTokenizer)) \
            and not token.startswith('▁'):
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


def adversarial_train(args, trainer, model=None, batch=None, k=3):
    # for pgd, we current hard code K as 3
    # TODO: add argument to allow change K for PGD method
    if args.adversarial_training_method == "fgm":
        trainer.attack()
        _, _, loss_adv = model(**batch)
        loss_adv.backward()
        trainer.restore()
    elif args.adversarial_training_method == "pgd":
        trainer.backup_grad()
        for t in range(k):
            trainer.attack(is_first_attack=(t == 0))
            if t != k - 1:
                model.zero_grad()
            else:
                trainer.restore_grad()
            _, _, loss_adv = model(**batch)
            loss_adv.backward()
        trainer.restore()
    else:
        raise RuntimeError(
            f"adopt adversarial training but use an unrecognized method name: {args.adversarial_training_method}")


def train(args, model, train_features, dev_features):
    """NER model training on train dataset; select model based on performance on dev dataset"""
    # create data loader
    data_loader = ner_data_loader(train_features, batch_size=args.train_batch_size, task='train', auto=True)
    # total training step counts
    t_total = len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # parameters for optimization
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # using fp16 for training rely on Nvidia apex package
    # fp16 training: try to use PyTorch naive implementation if available; we will only support apex anymore
    scaler = None
    autocast = None
    if args.fp16:
        try:
            autocast = torch.cuda.amp.autocast
            scaler = torch.cuda.amp.GradScaler()
        except Exception:
            raise ImportError("You need to update to PyTorch 1.6, the current PyTorch version is {}"
                              .format(torch.__version__))

    # training linear warm-up setup
    scheduler = None
    if args.do_warmup:
        warmup_steps = np.dtype('int64').type(args.warmup_ratio * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, min_lr=args.min_lr, num_training_steps=t_total)

    args.logger.info("***** Running training *****")
    args.logger.info("  Num data points = {}".format(len(data_loader)))
    args.logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.train_batch_size))
    args.logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    args.logger.info("  Total optimization steps = {}".format(t_total))
    args.logger.info("  Training steps (number of steps between two evaluation on dev) = {}".format(
        args.train_steps * args.gradient_accumulation_steps))
    args.logger.info("******************************")

    # create directory to save model
    new_model_dir = Path(args.new_model_dir)
    new_model_dir.mkdir(parents=True, exist_ok=True)
    # save label2idx json in new model directory
    json_dump(args.label2idx, new_model_dir / "label2idx.json")

    # save base model name to a base_model_name.txt
    with open(new_model_dir / "base_model_name.txt", "w") as f:
        f.write('model_type: {}\nbase_model: {}\nconfig: {}\ntokenizer: {}'.format(
            args.model_type, args.pretrained_model, args.config_name, args.tokenizer_name))

    global_step = 0
    tr_loss = .0
    best_score, epcoh_best_score = .0, .0
    early_stop_flag = 0

    model.zero_grad()

    # apply ADVERSARIAL TRAINING
    adversarial_trainer = ADVERSARIAL_TRAINER[args.adversarial_training_method](model) \
        if args.adversarial_training else None

    epoch_iter = trange(int(args.num_train_epochs), desc="Epoch", disable=not args.progress_bar)
    for epoch in epoch_iter:
        batch_iter = tqdm(iterable=data_loader, desc='Batch', disable=not args.progress_bar)
        for step, batch in enumerate(batch_iter):
            model.train()

            batch = tuple(b.to(args.device) for b in batch)
            train_inputs = batch_to_model_inputs(batch, args.model_type)

            if args.fp16:
                with autocast():
                    _, _, loss = model(**train_inputs)
            else:
                _, _, loss = model(**train_inputs)

            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # apply ADVERSARIAL TRAINING
            if args.adversarial_training:
                adversarial_train(args, adversarial_trainer, model, train_inputs)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.do_warmup:
                    scheduler.step()

                model.zero_grad()
                global_step += 1

            # using training step
            if args.train_steps > 0 and (global_step + 1) % args.train_steps == 0 and epoch > 0:
                # the current implementation will skip the all evaluations in the first epoch
                best_score, eval_loss = evaluate(
                    args, model, new_model_dir, dev_features, epoch, global_step, best_score)
                args.logger.info("""
                Global step: {}; 
                Epoch: {}; 
                average_train_loss: {:.4f}; 
                eval_loss: {:.4f}; 
                current best score: {:.4f}""".format(
                    global_step, epoch + 1, round(tr_loss / global_step, 4), eval_loss, best_score))

        # default model select method using strict F1-score with beta=1; evaluate model after each epoch on dev
        if args.train_steps <= 0 or epoch == 0:
            best_score, eval_loss = evaluate(
                args, model, new_model_dir, dev_features, epoch, global_step, best_score)
            args.logger.info("""
                Global step: {}; 
                Epoch: {}; 
                average_train_loss: {:.4f}; 
                eval_loss: {:.4f}; 
                current best score: {:.4f}""".format(
                global_step, epoch + 1, round(tr_loss / global_step, 4), eval_loss, best_score))

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

    # not data for evaluation
    if eval_size < 1:
        return [], [], .0

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
        eval_inputs = batch_to_model_inputs(batch, args.model_type)

        with torch.no_grad():
            raw_logits, _, loss = model(**eval_inputs)
            # get softmax output of the raw logits (keep dimensions)
            if not args.use_crf:
                raw_logits = torch.argmax(F.log_softmax(raw_logits, dim=-1), dim=-1)
            raw_logits = raw_logits.detach().cpu().numpy()
            # update evaluate loss
            eval_loss += loss.item()

        assert guards.shape == original_tkid.shape == original_mask.shape == original_labels.shape == raw_logits.shape, \
            """
                expect same dimension for all the inputs and outputs but get
                input_tokens: {}
                mask: {}
                label: {}
                logits: {}
            """.format(original_tkid.shape, original_mask.shape, original_labels.shape, raw_logits.shape)

        # tk=token, mk=mask, lb=label, lgt=logits
        for mks, lbs, lgts, gds in zip(original_mask, original_labels, raw_logits, guards):
            connect_sent_flag = False
            for mk, lb, lgt, gd in zip(mks, lbs, lgts, gds):
                if mk == 0:
                    # after hit first mask, we can stop for the current sentence since all rest will be pad
                    if args.model_type == "xlnet":
                        continue
                    else:
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

    return y_trues, y_preds, round(eval_loss / eval_size, 4)


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

    # select model based on best score
    # if best_score < cur_score:
    if cur_score - best_score > 1e-5:
        args.logger.info('''
        Global step: {}; 
        Epoch: {}; 
        previous best score: {:.4f}; 
        new best score: {:.4f}; 
        full evaluation metrix: {}
        '''.format(global_step, epoch + 1, best_score, cur_score, eval_metrix))
        best_score = cur_score
        save_model(args, model, new_model_dir, global_step, latest=args.max_num_checkpoints)

        # save model transformer core
        if args.save_model_core:
            save_only_transformer_core(args, model)

    return best_score, eval_loss


def __fix_bio(bios):
    fix_bios = []
    prev = None
    for i, bio in enumerate(bios):
        if i == 0:
            if bio.startswith("I-"):
                prev = "B-" + bio.split("-")[-1]
            else:
                prev = bio
        else:
            if bio.startswith("I-"):
                t, s = bio.split("-")
                if prev == "O":
                    prev = "B-" + s
                else:
                    pt, ps = prev.split("-")
                    if ps != s:
                        prev = "B-" + s
                    else:
                        prev = bio
            else:
                prev = bio
        fix_bios.append(prev)

    return fix_bios


def predict(args, model, features):
    """evaluation test dataset and output predicted results as BIO"""
    _, y_pred, _ = _eval(args, model, features)
    # fix prediction of system labels (X, SEP, CLS etc.)
    system_labels = {label for idx, label in args.idx2label.items() if idx < args.label2idx['O']}
    fixed_preds = []
    for each in y_pred:
        fixed_pred = list(map(lambda x: 'O' if x in system_labels else x, each))
        fixed_pred = __fix_bio(fixed_pred)  # fix case: O, i-x, i-x, O to O, b-x, i-x, O
        fixed_preds.append(fixed_pred)

    return fixed_preds


def _output_bio(args, tests, preds, save_bio=True):
    new_sents = []
    assert len(tests) == len(preds), "Expect {} sents but get {} sents in prediction".format(len(tests), len(preds))
    for example, predicted_labels in zip(tests, preds):
        tokens = example.text
        assert len(tokens) == len(predicted_labels), "Not same length sentence\nExpect: {} {}\nBut: {} {}".format(
            len(tokens), tokens, len(predicted_labels), predicted_labels)
        offsets = example.offsets
        if offsets:
            new_sent = [(tk, ofs[0], ofs[1], ofs[2], ofs[3], lb) for tk, ofs, lb in
                        zip(tokens, offsets, predicted_labels)]
        else:
            new_sent = [(tk, lb) for tk, lb in zip(tokens, predicted_labels)]
        new_sents.append(new_sent)
    if save_bio:
        output_bio(new_sents, args.predict_output_file)
        args.logger.warning("""The output file is {}, we recommend to use suffix as .bio.txt or .txt. 
                You can directly use formatter coverter tool in this package.""".format(
            args.predict_output_file))
        return None
    else:
        return new_sents


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

    if os.path.exists(args.new_model_dir) and os.listdir(args.new_model_dir) \
            and args.do_train and not args.overwrite_model_dir:
        raise ValueError(
            """new model directory: {} exists. 
            Use --overwrite_model_dir to overwrite the previous model. 
            Or create another directory for the new model""".format(args.new_model_dir))

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

    # if use resume_from_model,
    # we need to check the label in new data exact same as the one used in training previous model
    if args.do_train and args.resume_from_model is not None:
        label2idx_from_old = json_load(Path(args.resume_from_model / "label2idx.json"))
        assert len(label2idx_from_old) == len(label2idx), """expect same label2idx but get old one from resume model as {};
        and the new one from current data is {}""".format(label2idx_from_old, label2idx)
        for k in label2idx.keys():
            assert k in label2idx_from_old, f"the label {k} is not in old label2idx " \
                                            "check your data make sure all annotations are the same in two datasets"
        warnings.warn("will overwrite label2idx with label2idx from old model to make sure labels are mapped correct.")
        label2idx = label2idx_from_old

    num_labels = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}
    args.num_labels = num_labels
    args.label2idx = label2idx
    args.idx2label = idx2label

    # get config, model and tokenizer
    model_config, model_model, model_tokenizer = MODEL_CLASSES[args.model_type]
    args.logger.info("Training/evaluation parameters: {}".format({k: v for k, v in vars(args).items()}))

    # training
    if args.do_train:
        if args.model_type in {"roberta", "bart", "longformer", "deberta"}:
            # we need to set add_prefix_space to True for roberta, longformer, and Bart (any tokenizer from BPE)
            tokenizer = model_tokenizer.from_pretrained(
                args.tokenizer_name, do_lower_case=args.do_lower_case, add_prefix_space=True)
        else:
            tokenizer = model_tokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

        if args.resume_from_model is None:
            tokenizer.add_tokens(NEXT_TOKEN)
            config = model_config.from_pretrained(args.config_name, num_labels=num_labels)
            config.use_crf = args.use_crf
            config.label2idx = args.label2idx
            config.use_focal_loss = args.focal_loss
            config.focal_loss_gamma = args.focal_loss_gamma
            config.mlp_dim = args.mlp_dim
            args.logger.info("New Model Config:\n{}".format(config))
        else:
            config = model_config.from_pretrained(args.config_name, num_labels=num_labels)

        if args.resume_from_model is not None:
            args.config = config
            model = load_model(args, args.resume_from_model)
        else:
            model = model_model.from_pretrained(args.pretrained_model, config=config)
            # #add an control token for combine sentence if it is too long to fit max_seq_len
            model.resize_token_embeddings(len(tokenizer))
            config.vocab_size = len(tokenizer)
            args.config = model.config

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
        # set up ADVERSARIAL TRAINING training
        args.adversarial_training = True if args.adversarial_training_method is not None else False
        # start training
        train(args, model, train_features, dev_features)
        # save config and tokenizer with new model
        args.tokenizer.save_pretrained(args.new_model_dir)
        args.config.save_pretrained(args.new_model_dir)

    # predict - test.txt file prediction (if you need predict many files, use 'run_transformer_batch_prediction')
    if args.do_predict:
        args.config = model_config.from_pretrained(args.new_model_dir, num_labels=num_labels)
        # args.use_crf = args.config.use_crf
        # args.model_type = args.config.model_type
        if args.model_type in {"roberta", "bart", "longformer", "deberta"}:
            # we need to set add_prefix_space to True for roberta, longformer, and Bart (any tokenizer from GPT-2)
            tokenizer = model_tokenizer.from_pretrained(
                args.new_model_dir, do_lower_case=args.do_lower_case, add_prefix_space=True)
        else:
            args.tokenizer = model_tokenizer.from_pretrained(args.new_model_dir, do_lower_case=args.do_lower_case)
        model = load_model(args)
        model.to(args.device)

        test_example = ner_data_processor.get_test_examples()
        test_features = transformer_convert_data_to_features(args,
                                                             input_examples=test_example,
                                                             label2idx=label2idx,
                                                             tokenizer=args.tokenizer,
                                                             max_seq_len=args.max_seq_length)

        predictions = predict(args, model, test_features)
        _output_bio(args, test_example, predictions)
