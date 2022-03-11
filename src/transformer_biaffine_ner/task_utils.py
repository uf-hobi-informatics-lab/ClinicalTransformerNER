# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21
import traceback

import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from common_utils.common_io import json_dump, write_to_file
from transformer_ner.model_utils import PGD, FGM, get_linear_schedule_with_warmup
from transformer_biaffine_ner.model import TransformerBiaffineNerModel
from transformer_biaffine_ner.data_utils import batch_to_model_inputs
from transformers import AutoTokenizer, AutoConfig, get_constant_schedule_with_warmup
from transformer_ner.task import MODEL_CLASSES


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
    sub_indexes = []
    eval_loss = .0
    total_samples = len(data_loader)

    model.eval()
    with torch.no_grad():
        batch_iter = tqdm(iterable=data_loader, desc='evaluation', disable=not args.progress_bar)
        for step, batch in enumerate(batch_iter):
            token_ids = batch[0].numpy()
            sub_index = batch[5].numpy()
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
            sub_indexes.extend(sub_index)

    return y_trues, y_preds, tok_ids, sub_indexes, round(eval_loss / total_samples, 4)


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

    y_trues, y_preds, _, _, loss = _get_predictions(args, model, data_loader)

    precision, recall, f1 = _get_eval_metrics(y_trues, y_preds)

    # save model if eval score is better (at least 1e-5 improvement)
    new_best_score = prev_best_score
    if f1 - prev_best_score > 1e-6:
        args.logger.info('''
        previous best score: {:.4f}; 
        new best score: {:.4f}; 
        '''.format(prev_best_score, f1))
        new_best_score = f1
        save_model(args, model, new_model_dir, global_step, latest=args.max_num_checkpoints)

    return new_best_score, f1, precision, recall, loss


# def _get_decode_mapping(tok_ids, cur_index, query_word, tokenizer=None):
#     while cur_index < len(tok_ids):
#         j = cur_index + 1
#         while j <= len(tok_ids):
#             en = tokenizer.decode(tok_ids[cur_index: j], clean_up_tokenization_spaces=False).strip()
#             if en == query_word:
#                 return cur_index, j
#             j += 1
#         cur_index += 1
#
#     raise RuntimeError(f"Cannot map indexes for {query_word} in"
#                        f"\n{tokenizer.decode(tok_ids, skip_special_tokens=True).strip().split()}")


def _decode_index_mapping(map_table, s, e):
    new_s = None
    new_e = None
    for k, v in map_table.items():
        range_s, range_e = v
        if range_s <= s <= range_e:
            new_s = k
        if range_s <= e <= range_e:
            new_e = k
    return new_s, new_e


def predict(args, data_loader):
    args.logger.info("***** Running evaluation on {} number of test data *****".format(len(data_loader)))
    args.logger.info("  Instantaneous batch size per GPU = {}".format(args.eval_batch_size))
    args.logger.info("******************************")

    # set up model
    model = load_model(args)
    model.to(args.device)

    _, preds, tok_ids, sub_indexes, _ = _get_predictions(args, model, data_loader)
    assert len(preds) == len(tok_ids) == len(sub_indexes), \
        f"pred: {len(preds)} sentences; tokens: {len(tok_ids)} sentences; indexes: {len(sub_indexes)} sentences"

    predicted_outputs = []
    for pred, tok_id, sub_index in tqdm(zip(preds, tok_ids, sub_indexes), total=len(tok_ids), desc="decoding"):
        output = []

        # using sub_index to create index remap
        indexes_remap = dict()
        for i, idx in enumerate(sub_index):
            if idx == 0:
                continue
            # idx start with special token, we need to subtract 1 to re-align to original index
            org_idx = idx - 1
            if org_idx in indexes_remap:
                indexes_remap[org_idx][1] = i
            else:
                indexes_remap[org_idx] = [i, i]

        sent_text = args.tokenizer.decode(tok_id, skip_special_tokens=True)

        for each in pred:
            en_type_id, s, e = each
            en_text = args.tokenizer.decode(tok_id[s: e + 1], clean_up_tokenization_spaces=False)

            new_s, new_e = _decode_index_mapping(indexes_remap, s, e)

            # in case s, e cannot be mapped
            if new_s is None or new_e is None:
                args.logger.warning(f"cannot decode entity at ({s}, {e})\nentity: {en_text}\nsentence: {sent_text}")
                continue

            # for list slice
            new_e += 1

            en_type = args.config.idx2label[en_type_id]

            output.append((en_type, int(new_s), int(new_e), en_text))

        # note: sent_text and en_text cannot be used for sanity check / only for visual check
        predicted_outputs.append({"tokens": sent_text, "entities": output})

        #################################################################################################
        # # due to inconsistent behavior of decode function, the following decoding is not working correctly
        # sent_text = args.tokenizer.decode(
        #     tok_id, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().split(" ")

        # build a token remap to map tok ids positions to token positions
        # indexes_remap = dict()
        # cur_index = 0
        # for i, word in enumerate(sent_text):
        #     word = word.strip()
        #     cur_index, new_index = _get_decode_mapping(tok_id, cur_index, query_word=word, tokenizer=args.tokenizer)
        #     indexes_remap[i] = (cur_index, new_index)
        #     cur_index = new_index

        # for each in pred:
        #     en_type_id, s, e = each
        #     en = args.tokenizer.decode(tok_id[s: e + 1], clean_up_tokenization_spaces=False).strip()
        #     new_s, new_e = _decode_index_mapping(indexes_remap, s, e)
        #     new_e += 1
        #     # check if decode is right
        #     en_check = " ".join(sent_text[new_s: new_e])
        #     if en != en_check:
        #         args.logger.warning(f"decode error: expect: {en} but get "
        #                             f"{en_check}\n"
        #                             f"{[e for e in tok_id if e != args.tokenizer.pad_token_id]}\n"
        #                             f"{sent_text}\n{each}")
        #
        #     en_type = args.config.idx2label[int(en_type_id)]
        #     output.append((en, en_type, new_s, new_e))
        # predicted_outputs.append({"tokens": sent_text, "entities": output})
        #################################################################################################

    return predicted_outputs


def train(args, train_data_loader, dev_data_loader):
    # setup model
    model = TransformerBiaffineNerModel(args.config)
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
            global_step += 1

            # using training step
            if args.train_steps > 0 and (global_step + 1) % args.train_steps == 0 and args.epoch > 0:
                # the current implementation will skip the all evaluations in the first epoch
                best_score, f1, pre, rec, eval_loss = _evaluate(
                    args, best_score, model, new_model_dir, global_step, dev_data_loader)

                args.logger.info(_eval_info(args, global_step, epoch, eval_loss, pre, rec, f1, best_score))

        if args.train_steps <= 0 or epoch == 0:
            best_score, f1, pre, rec, eval_loss = _evaluate(
                args, best_score, model, new_model_dir, global_step, dev_data_loader)

            args.logger.info(_eval_info(args, global_step, epoch, eval_loss, pre, rec, f1, best_score))

        # early stop check
        if best_score - epcoh_best_score > 1e-5:
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
    args.tr_loss += loss.item()

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


def _eval_info(args, global_step, epoch, eval_loss, pre, rec, f1, best_score):
    info = """
        Global step: {}; 
        Epoch: {}; 
        average_train_loss: {:.4f}; 
        eval_loss: {:.4f};
        precision: {:.4f};
        recall: {:.4f};
        f1: {:.4f};
        current best F1 score: {:.4f}""".format(
        global_step, epoch + 1,
        round(args.tr_loss / global_step, 4),
        eval_loss,
        pre,
        rec,
        f1,
        best_score)
    return info


def get_tokenizer(args, is_train=True):
    if is_train:
        tokenizer_path = args.tokenizer_name
    else:
        tokenizer_path = args.new_model_dir

    # # AutoTokenizer has a problem with BERT case, we will use MODEL_CLASSES for now until issue fixed
    # if args.model_type in {"roberta", "bart", "longformer", "deberta"}:
    #     # we need to set add_prefix_space to True for roberta, longformer, and Bart (any tokenizer from BPE)
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         tokenizer_path, do_lower_case=args.do_lower_case, add_prefix_space=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)

    # # alternative to get tokenizer
    _, _, tokenizer_init = MODEL_CLASSES[args.model_type]
    if args.model_type in {"roberta", "bart", "longformer", "deberta"}:
        # we need to set add_prefix_space to True for roberta, longformer, and Bart (any tokenizer from BPE)
        tokenizer = tokenizer_init.from_pretrained(
            tokenizer_path, do_lower_case=args.do_lower_case, add_prefix_space=True)
    else:
        tokenizer = tokenizer_init.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)

    return tokenizer


def get_config(args, is_train=True):
    if is_train:
        config_path = args.config_name
        config = AutoConfig.from_pretrained(config_path, num_labels=args.num_classes)
    else:
        config_path = args.new_model_dir
        config = AutoConfig.from_pretrained(config_path)
    return config


def save_model(args, model, model_dir, index, latest=3):
    new_model_file = model_dir / "checkpoint_{}.bin".format(index)
    args.tokenizer.save_pretrained(args.new_model_dir)
    args.config.save_pretrained(args.new_model_dir)

    # save model as state dict
    args.logger.info("save checkpoint to {}".format(new_model_file))
    torch.save(model.state_dict(), new_model_file)

    # only keep 'latest' # of checkpoints
    all_model_files = list(model_dir.glob("checkpoint_*.bin"))
    if len(all_model_files) > latest:
        # sorted_files = sorted(all_model_files, key=lambda x: os.path.getmtime(x))  # sorted by the last modified time
        sorted_file = sorted(all_model_files, key=lambda x: int(x.stem.split("_")[-1]))  # sorted by the checkpoint step
        file_to_remove = sorted_file[0]  # remove earliest checkpoints
        file_to_remove.unlink()


def load_model(args):
    model_dir = Path(args.new_model_dir)

    all_model_files = model_dir.glob("*.bin")
    all_model_files = list(filter(lambda x: 'checkpoint_' in x.as_posix(), all_model_files))
    sorted_file = sorted(all_model_files, key=lambda x: int(x.stem.split("_")[-1]))

    # #load direct from model files
    args.logger.info("load checkpoint from {}".format(sorted_file[-1]))
    ckpt = torch.load(sorted_file[-1], map_location=torch.device('cpu'))

    try:
        model = TransformerBiaffineNerModel
        model = model(config=args.config)
        model.load_state_dict(state_dict=ckpt)
        return model
    except AttributeError as Ex:
        args.logger.error(traceback.format_exc())
        raise RuntimeError("""You need to check the model save/load processes.""")
