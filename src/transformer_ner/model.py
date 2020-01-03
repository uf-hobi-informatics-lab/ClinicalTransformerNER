#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
We tried to implement a common class BertLikeNER for BERT, ROBERTA, ALBERT, DISTILBERT to share the common forward() function;
However, such implementation will dramatically influence the model converage process.
The current implementation has repeated code but will guarantee the performance for each model.
"""

from transformers import (BertConfig,  BertModel, BertPreTrainedModel,
                          RobertaModel, RobertaConfig, PreTrainedModel,
                          XLNetModel, XLNetPreTrainedModel, XLNetConfig,
                          AlbertModel, AlbertConfig,
                          DistilBertConfig, DistilBertModel,
                          ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                          DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                          XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
                          ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                          BERT_PRETRAINED_MODEL_ARCHIVE_MAP)
from torch import nn


class BertLikeNerModel(PreTrainedModel):
    """not fit for the current training; but can be integrated into new APP"""
    CONF_REF = {
        'bert': (BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, 'bert'),
        'roberta': (RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, 'roberta'),
        'albert': (AlbertConfig, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP, 'albert')
    }

    def __init__(self, config, model_type):
        super().__init__(config)
        self.model_type = model_type
        self.num_labels = config.num_labels
        self.model = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.__prepare_model_instance(config)
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def __prepare_model_instance(self, config):
        self.config_class, self.pretrained_model_archive_map, self.base_model_prefix = self.CONF_REF[self.model_type]
        if self.model_type == "bert":
            self.model = BertModel(config)
        elif self.model_type == 'roberta':
            self.model = RobertaModel(config)
        elif self.model_type == 'albert':
            self.model = AlbertModel(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            raise NotImplementedError('CRFs layer is not supported yet')
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class BertNerModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            raise NotImplementedError('CRFs layer is not supported yet')
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class RobertaNerModel(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        """
        :return: raw logits without any softmax or log_softmax transformation

        qoute for reason (https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/7):
        You should pass raw logits to nn.CrossEntropyLoss, since the function itself applies F.log_softmax and nn.NLLLoss() on the input.
        If you pass log probabilities (from nn.LogSoftmax) or probabilities (from nn.Softmax()) your loss function won’t work as intended.

        From the pytorch CrossEntropyLoss doc:
        The input is expected to contain raw, unnormalized scores for each class.

        If apply CRF, we cannot use CrossEntropyLoss but instead using NLLLoss ()
        """
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        seq_outputs = outputs[0]
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)

        if self.use_crf:
            raise NotImplementedError('CRFs layer is not supported yet')
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class AlbertNerModel(BertPreTrainedModel):
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'albert'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        outputs = self.albert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask)

        seq_outputs = outputs[0]
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)

        if self.use_crf:
            raise NotImplementedError('CRFs layer is not supported yet')
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class DistilBertNerModel(BertPreTrainedModel):
    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'distilbert'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        outputs = self.distilbert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            raise NotImplementedError('CRFs layer is not supported yet')
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class XLNetNerModel(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.xlnet = XLNetModel(config)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids, attention_mask=None, mems=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        outputs = self.xlnet(input_ids,
                             attention_mask=attention_mask,
                             mems=mems,
                             token_type_ids=token_type_ids,
                             head_mask=head_mask)

        seq_outputs = outputs[0]
        logits = self.classifier(seq_outputs)

        if self.use_crf:
            raise NotImplementedError('CRFs layer is not supported yet')
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss
