# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from common_utils.common_io import json_dump, write_to_file
from transformer_ner.model_utils import PGD, FGM, get_linear_schedule_with_warmup
from transformer_ner.task import save_model, load_model
from transformer_biaffine_ner.model import TransformerBiaffineNerModel
from transformer_biaffine_ner.data_utils import batch_to_model_inputs
from transformers import AutoTokenizer, AutoConfig, get_constant_schedule_with_warmup


def _get_label_from_span(span_in_batch):
    assert len(span_in_batch.shape) == 3
    results = []
    for span in span_in_batch:
        result = []
        for indexes in np.argwhere(span):
            s, e = indexes
            en_type_id = span[s, e]
            result.append((en_type_id, s, e))
        results.append(result)

    return results


def _get_predictions(args, model, data_loader):
    y_trues, y_preds = [], []
    tok_ids = []
    eval_loss = .0
    total_samples = len(data_loader)

    model.eval()
    with torch.no_grad():
        batch_iter = tqdm(iterable=data_loader, desc='evaluation', disable=not args.progress_bar)
        for step, batch in enumerate(batch_iter):
            token_ids = batch[0].numpy()
            labels = batch[3].numpy()
            label_masks = batch[4].numpy()

            batch = batch_to_model_inputs(batch, device=args.device)
            logits, loss = model(**batch)
            eval_loss += loss.item()
            preds = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1).detach().cpu().numpy()
            # keep label masked places as 0
            preds *= label_masks

            assert preds.shape == labels.shape, "predicted outputs have different dim with gold standards."

            # we convert predictions and labels to tuples e.g., (type, s, e)
            y_true = _get_label_from_span(labels)
            y_pred = _get_label_from_span(preds)

            y_trues.extend(y_true)
            y_preds.extend(y_pred)
            tok_ids.extend(token_ids)

    return y_trues, y_preds, tok_ids, round(eval_loss/total_samples, 4)


def _get_eval_metrics(labels, preds):
    assert len(labels) == len(preds), f"pred: {len(preds)} sentences but gold standards has {len(labels)} sentences"
    # simplify evaluation as a binary f1 score measurement
    tp, fp, fn = 0, 0, 0
    for label, pred in zip(labels, preds):
        label = set(label)
        pred = set(pred)
        commons = label & pred
        tp += len(commons)
        fp += len(pred - commons)
        fn += len(label - commons)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1


def _evaluate(args, prev_best_score, model, new_model_dir, global_step, data_loader):
    args.logger.info("***** Running evaluation on {} number of dev data *****".format(len(data_loader)))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.eval_batch_size))
    args.logger.info("******************************")

    y_trues, y_preds, _, loss = _get_predictions(args, model, data_loader)

    precision, recall, f1 = _get_eval_metrics(y_trues, y_preds)

    # save model if eval score is better (at least 1e-5 improvement)
    new_best_score = prev_best_score
    if f1 - prev_best_score > 5e-6:
        args.logger.info('''
        previous best score: {:.4f}; 
        new best score: {:.4f}; 
        '''.format(prev_best_score, f1))
        new_best_score = f1
        save_model(args, model, new_model_dir, global_step, latest=args.max_num_checkpoints)

    return new_best_score, precision, recall, loss


def predict(args, data_loader):
    args.logger.info("***** Running evaluation on {} number of dev data *****".format(len(data_loader)))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.eval_batch_size))
    args.logger.info("******************************")

    # set up model
    model = load_model(args.new_model_dir)
    model.to(args.device)

    _, preds, tok_ids, _ = _get_predictions(args, model, data_loader)
    assert len(preds) == len(tok_ids), \
        f"pred: {len(preds)} sentences but tokens has {len(tok_ids)} sentences"

    predicted_outputs = []
    for pred, tok_id in zip(preds, tok_ids):
        output = []
        for each in pred:
            en_type_id, s, e = each
            en = args.tokenizer.decode(tok_id[s:e+1]).strip()
            ll = len(en.split())
            new_s = s
            new_e = s + ll  # note we include extra pos here; we do not need to do [s: e+1] instead use [s:e] later
            output.append((en, en_type_id, new_s, new_e))
        sent_text = args.tokenizer.decode(tok_id).strip()
        predicted_outputs.append({"tokens": sent_text.split(), "entities": output})

    return predicted_outputs


def train(args, train_data_loader, dev_data_loader):
    # setup model
    model = TransformerBiaffineNerModel(args.config, args.pretrained_model)
    model.resize_token_embeddings(args.config.vocab_size)
    model.to(args.device)

    total_steps = len(train_data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.optimizer = _get_optimizer(args, model)
    args.scheduler = _get_scheduler(args, total_steps=total_steps)
    args.scaler, args.autocast = _fp16(args)

    # create directory to save model
    new_model_dir = Path(args.new_model_dir)
    new_model_dir.mkdir(parents=True, exist_ok=True)
    # save label2idx json in new model directory
    json_dump(args.label2idx, new_model_dir / "label2idx.json")
    # save model information
    write_to_file(
        f"base model: {args.model_type}\n pretrained_model: {args.pretrained_model}",
        new_model_dir / "base_model_name.txt"
    )

    _print_info(args, train_data_loader, total_steps)

    global_step, args.tr_loss = 0, .0
    best_score, epcoh_best_score = .0, .0
    early_stop_flag = 0

    model.zero_grad()
    epoch_iter = trange(int(args.num_train_epochs), desc="Epoch", disable=not args.progress_bar)
    for epoch in epoch_iter:
        args.epoch = epoch
        batch_iter = tqdm(iterable=train_data_loader, desc='Batch', disable=not args.progress_bar)
        for step, batch in enumerate(batch_iter):
            _train_step(args, model, batch, step)
            args.global_step += 1

            # using training step
            if args.train_steps > 0 and (args.global_step + 1) % args.train_steps == 0 and args.epoch > 0:
                # the current implementation will skip the all evaluations in the first epoch
                best_score, pre, rec, eval_loss = _evaluate(
                    args, best_score, model, new_model_dir, global_step, dev_data_loader)

                args.logger.info("""
                                Global step: {}; 
                                Epoch: {}; 
                                average_train_loss: {:.4f}; 
                                eval_loss: {:.4f};
                                precision: {:.4f};
                                recall: {:.4f};
                                current best F1 score: {:.4f}""".format(
                    args.global_step, epoch + 1,
                    round(args.tr_loss / args.global_step, 4),
                    eval_loss,
                    pre,
                    rec,
                    args.best_score))

        if args.train_steps <= 0 or epoch == 0:
            best_score, pre, rec, eval_loss = _evaluate(
                args, best_score, model, new_model_dir, global_step, dev_data_loader)

            args.logger.info("""
                            Global step: {}; 
                            Epoch: {}; 
                            average_train_loss: {:.4f}; 
                            eval_loss: {:.4f};
                            precision: {:.4f};
                            recall: {:.4f};
                            current best F1 score: {:.4f}""".format(
                args.global_step, epoch + 1,
                round(args.tr_loss / args.global_step, 4),
                eval_loss,
                pre,
                rec,
                args.best_score))

        # early stop check
        if best_score - args.epcoh_best_score > 1e-5:
            epcoh_best_score = best_score
            early_stop_flag = 0
        else:
            early_stop_flag += 1
        if 0 < args.early_stop <= early_stop_flag:
            args.logger.warning('Early stop activated; performance not improve anymore.')
            break

        batch_iter.reset()
    epoch_iter.reset()


def _train_step(args, model, current_batch, current_step):
    model.train()

    train_inputs = batch_to_model_inputs(current_batch, device=args.device)

    if args.fp16:
        with args.autocast():
            _, loss = model(**train_inputs)
    else:
        _, loss = model(**train_inputs)

    loss = loss / args.gradient_accumulation_steps
    args.train_loss += loss.item()

    if args.fp16:
        args.scaler.scale(loss).backward()
    else:
        loss.backward()

    if (current_step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            args.scaler.unscale_(args.optimizer)
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            args.scaler.step(args.optimizer)
            args.scaler.update()
        else:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            args.optimizer.step()

        args.scheduler.step()
        model.zero_grad()


def _fp16(args):
    if args.fp16:
        try:
            autocast = torch.cuda.amp.autocast
            scaler = torch.cuda.amp.GradScaler()
            return scaler, autocast
        except Exception:
            args.logger.error(
                "You need to update to PyTorch 1.6, the current PyTorch version is {}".format(torch.__version__))
            args.logger.warning("Set fp16 to False, continue with fp32.")
            args.fp16 = False
            return None, None
    else:
        return None, None


def _get_optimizer(args, model):
    # parameters for optimization
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    return optimizer


def _get_scheduler(args, total_steps=1e5):
    if args.do_warmup:
        # if do warm up we actually do linear schedule
        warmup_steps = np.dtype('int64').type(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            args.optimizer, num_warmup_steps=warmup_steps, min_lr=args.min_lr, num_training_steps=total_steps)
    else:
        # we still do warmup but lr will remain after warmup
        warmup_steps = np.dtype('int64').type(args.warmup_ratio * total_steps)
        scheduler = get_constant_schedule_with_warmup(args.optimizer, num_warmup_steps=warmup_steps)

    return scheduler


def _print_info(args, data_loader, total_steps):
    args.logger.info("***** Running training with biaffine mode *****")
    args.logger.info("  total data points = {}".format(len(data_loader)))
    args.logger.info("  total Epochs = {}".format(args.num_train_epochs))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.train_batch_size))
    args.logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    args.logger.info("  Total optimization steps = {}".format(total_steps))
    args.logger.info("  Training steps (number of steps between two evaluation on dev) = {}".format(
        args.train_steps * args.gradient_accumulation_steps))
    args.logger.info("******************************")


def get_tokenizer(args, is_train=True):
    if is_train:
        tokenizer_path = args.tokenizer_name
    else:
        tokenizer_path = args.new_model_dir

    if args.model_type in {"roberta", "bart", "longformer", "deberta"}:
        # we need to set add_prefix_space to True for roberta, longformer, and Bart (any tokenizer from BPE)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=args.do_lower_case, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)

    return tokenizer


def get_config(args, is_train=True):
    if is_train:
        config_path = args.config_name
    else:
        config_path = args.new_model_dir

    config = AutoConfig.from_pretrained(config_path, num_labels=args.num_labels)

    return config
