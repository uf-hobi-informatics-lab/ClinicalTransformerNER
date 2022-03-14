import argparse

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

from common_utils.common_io import json_load

from config import Config
import data_loader
import utils
from model import Model


class Trainer(object):
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * self.config.updates_total,
                                                                      num_training_steps=self.config.updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels)
            pred_result.append(outputs)

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="micro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        self.config.logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="micro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        self.config.logger.info('{} Label F1 {}'.format(title, f1_score(label_result.cpu().numpy(),
                                                            pred_result.cpu().numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        self.config.logger.info("\n{}".format(table))
        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def load_config(args, config_fn):
    configs = json_load(config_fn)
    return Config(args, configs)


def run_task(args):
    # required to specify a config file and we load all related config from that file
    logger = args.logger
    config_file = args.w2ner_config
    config = load_config(args, config_file)

    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)



    logger.info("Loading Data")
    datasets = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    config.updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model, config)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if f1 > best_f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save("model.pt")
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load("model.pt")
    trainer.eval("Final", test_loader, True)
