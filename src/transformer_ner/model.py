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
                          BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                          BartConfig, BartModel, ElectraForTokenClassification,
                          ElectraModel, XLNetForTokenClassification, AlbertPreTrainedModel,
                          RobertaForTokenClassification)
from torch import nn
import torch


class Transformer_CRF(nn.Module):
    def __init__(self, num_labels, start_label_id):
        super().__init__()
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels), requires_grad=True)
        self.log_alpha = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.score = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.log_delta = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.psi = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.path = nn.Parameter(torch.zeros(1, 1, dtype=torch.long), requires_grad=False)

    @staticmethod
    def log_sum_exp_batch(log_Tensor, axis=-1):
        # shape (batch_size,n,m)
        return torch.max(log_Tensor, axis)[0] + \
               torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))

    def reset_layers(self):
        self.log_alpha = self.log_alpha.fill_(0.)
        self.score = self.score.fill_(0.)
        self.log_delta = self.log_delta.fill_(0.)
        self.psi = self.psi.fill_(0.)
        self.path = self.path.fill_(0)

    def forward(self, feats, label_ids):
        forward_score = self._forward_alg(feats)
        max_logLL_allz_allx, path, gold_score = self._crf_decode(feats, label_ids)
        loss = torch.mean(forward_score - gold_score)
        ####
        # print(forward_score)
        # print(gold_score)
        # print(max_logLL_allz_allx)
        # print(path)
        ####
        self.reset_layers()
        return path, max_logLL_allz_allx, loss

    def _forward_alg(self, feats):
        """alpha-recursion or forward recursion; to compute the partition function"""
        # feats -> (batch size, num_labels)
        seq_size = feats.shape[1]
        batch_size = feats.shape[0]
        log_alpha = self.log_alpha.expand(batch_size, 1, self.num_labels).clone().fill_(-10000.)
        log_alpha[:, 0, self.start_label_id] = 0
        for t in range(1, seq_size):
            log_alpha = (self.log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        return self.log_sum_exp_batch(log_alpha)

    def _crf_decode(self, feats, label_ids):
        seq_size = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)
        score = self.score.expand(batch_size, 1)

        log_delta = self.log_delta.expand(batch_size, 1, self.num_labels).clone().fill_(-10000.)
        log_delta[:, 0, self.start_label_id] = 0
        psi = self.psi.expand(batch_size, seq_size, self.num_labels).clone()

        for t in range(1, seq_size):
            score = score + \
                    batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t-1]).view(-1, 1)) + \
                    feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)

            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = self.path.expand(batch_size, seq_size).clone()
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)
        for t in range(seq_size-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path, score


class BertLikeNerModel(PreTrainedModel):
    """not fit for the current training; but can be integrated into new APP"""
    CONF_REF = {
        'bert': (BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, 'bert'),
        'roberta': (RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, 'roberta'),
        'albert': (AlbertConfig, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP, 'albert')
    }

    def __init__(self, config, model_type, crf=None):
        super().__init__(config)
        self.model_type = model_type
        self.num_labels = config.num_labels
        self.model = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = None
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
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            # loss_fct = nn.CrossEntropyLoss()  # CrossEntropyLoss has log_softmax operation inside
            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class BertNerModel(BertPreTrainedModel):
    """
    model architecture:
      (bert): BertModel
      (dropout): Dropout(p=0.1, inplace=False)
      (classifier): Linear(in_features=768, out_features=12, bias=True)
      (loss_fct): CrossEntropyLoss()
      (crf_layer): Transformer_CRF()
    """
    def __init__(self, config, crf=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = crf
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
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            # loss_fct = nn.CrossEntropyLoss()
            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class RobertaNerModel(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, crf=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = crf
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
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
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


class AlbertNerModel(AlbertPreTrainedModel):
    # config_class = AlbertConfig
    # pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    # base_model_prefix = 'albert'

    def __init__(self, config, crf=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = crf
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
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
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

    def __init__(self, config, crf=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = crf
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, label_ids=None):
        outputs = self.distilbert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
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


class XLNetNerModel(XLNetForTokenClassification):
    def __init__(self, config, crf=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.xlnet = XLNetModel(config)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = crf
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids=None,
                attention_mask=None,
                mems=None,
                perm_mask=None,
                target_mapping=None,
                token_type_ids=None,
                input_mask=None,
                head_mask=None,
                inputs_embeds=None,
                label_ids=None):
        outputs = self.xlnet(input_ids=input_ids,
                             attention_mask=attention_mask,
                             mems=mems,
                             perm_mask=perm_mask,
                             target_mapping=target_mapping,
                             token_type_ids=token_type_ids,
                             input_mask=input_mask,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds)

        seq_outputs = outputs[0]
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)

        if self.use_crf:
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
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


class BartNerModel(PreTrainedModel):
    """
        According to https://arxiv.org/pdf/1910.13461.pdf section 3.2,
        the token classification tasks use the top decoder hidden state.
        We will adopt their implementation only using the decoder (dco) for classification,
        we do provide the option to concat encoder output with decoder output.
    """
    config_class = BartConfig
    base_model_prefix = "bart"
    pretrained_model_archive_map = {"bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin"}

    def __init__(self, config, crf=None, output_concat=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bart = BartModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.output_concat = output_concat
        self.crf_layer = crf
        self.init_weights()

    def _init_weights(self, module):
        std = self.config.init_std
        # called init_bert_params in fairseq
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, encoder_outputs=None, decoder_attention_mask=None, decoder_cached_states=None, label_ids=None):
        # dco = decoder output; eco = encoder output
        dco, eco = self.bart(input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             encoder_outputs=encoder_outputs,
                             decoder_attention_mask=decoder_attention_mask,
                             decoder_cached_states=decoder_cached_states
                             )
        if self.output_concat:
            sequence_output = torch.cat((dco, eco), 2)
        else:
            sequence_output = dco

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            # loss_fct = nn.CrossEntropyLoss()
            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss


class ElectraNerModel(ElectraForTokenClassification):
    """
    model architecture:
      (bert): ELECTRA
      (dropout): Dropout(p=0.1, inplace=False)
      (classifier): Linear(in_features=768, out_features=12, bias=True)
      (loss_fct): CrossEntropyLoss()
      (crf_layer): Transformer_CRF()
    """

    def __init__(self, config, crf=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_crf = False
        self.crf_layer = crf
        self.init_weights()

    def active_using_crf(self):
        self.use_crf = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, label_ids=None):
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               inputs_embeds=inputs_embeds,
                               head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            logits, active_logits, loss = self.crf_layer(logits, label_ids)
        else:
            if attention_mask is not None:
                active_idx = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = label_ids.view(-1)[active_idx]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)

            # loss_fct = nn.CrossEntropyLoss()
            loss = self.loss_fct(active_logits, active_labels)

        return logits, active_logits, loss