#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The current implementation has repeated code but will guarantee the performance for each model.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
                          BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
                          DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
                          ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
                          XLNET_PRETRAINED_MODEL_ARCHIVE_LIST, AlbertConfig,
                          AlbertModel, AlbertPreTrainedModel, BartConfig,
                          BartModel, BertConfig, BertModel,
                          BertPreTrainedModel, DebertaModel,
                          DebertaPreTrainedModel, DistilBertConfig,
                          DistilBertModel, ElectraForTokenClassification,
                          ElectraModel, LongformerForTokenClassification,
                          LongformerModel, PreTrainedModel, RobertaConfig,
                          RobertaForTokenClassification, RobertaModel,
                          XLNetConfig, XLNetForTokenClassification, XLNetModel,
                          XLNetPreTrainedModel, DebertaV2Model, DebertaV2ForTokenClassification,
                          MegatronBertPreTrainedModel, MegatronBertModel, MegatronBertPreTrainedModel)

from transformer_ner.model_utils import FocalLoss, _calculate_loss
from transformer_ner.model_utils import New_Transformer_CRF as Transformer_CRF


class BertNerModel(BertPreTrainedModel):
    """
    model architecture:
      (bert): BertModel
      (dropout): Dropout(p=0.1, inplace=False)
      (classifier): Linear(in_features=768, out_features=12, bias=True)
      (loss_fct): CrossEntropyLoss()
      (crf_layer): Transformer_CRF()
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, label_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class RobertaNerModel(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, label_ids=None):
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
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class LongformerNerModel(LongformerForTokenClassification):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                global_attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                label_ids=None,
                output_attentions=None,
                output_hidden_states=None):
        outputs = self.longformer(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  global_attention_mask=global_attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=inputs_embeds,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class AlbertNerModel(AlbertPreTrainedModel):
    # config_class = AlbertConfig
    # pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    # base_model_prefix = 'albert'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, label_ids=None):
        outputs = self.albert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask)

        seq_outputs = outputs[0]
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class DistilBertNerModel(BertPreTrainedModel):
    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = 'distilbert'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                label_ids=None):

        outputs = self.distilbert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class XLNetNerModel(XLNetForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.xlnet = XLNetModel(config)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        if config.use_crf:
            raise Warning("Not support CRF for XLNet for now.")
        if config.use_biaffine:
            raise Warning("Not support biaffine for XLNet for now")
        # will not support crf and biaffine
        self.crf_layer = None
        self.biaffine = None
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                mems=None,
                perm_mask=None,
                target_mapping=None,
                token_type_ids=None,
                input_mask=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=True,
                label_ids=None,
                output_attentions=None,
                output_hidden_states=None,
        ):

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

        loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

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
    pretrained_model_archive_map = {
        "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin"}

    def __init__(self, config, output_concat=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bart = BartModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.output_concat = output_concat
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

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, encoder_outputs=None,
                decoder_attention_mask=None, decoder_cached_states=None, label_ids=None):
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
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

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

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                label_ids=None,
                output_attentions=None,
                output_hidden_states=None):

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
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class DeBertaNerModel(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            label_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class DeBertaV2NerModel(DebertaV2ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta_v2 = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            label_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        outputs = self.deberta_v2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class MegatronNerModel(MegatronBertPreTrainedModel):
    """
    model architecture:
      (bert): MegatronBertModel
      (dropout): Dropout(p=0.1, inplace=False)
      (classifier): Linear(in_features=768, out_features=12, bias=True)
      (loss_fct): CrossEntropyLoss()
      (crf_layer): Transformer_CRF()
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.use_crf = config.use_crf if hasattr(config, "use_crf") else None
        self.crf_layer = Transformer_CRF(config.num_labels) if self.use_crf else None

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, label_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.use_crf:
            # logits, active_logits, loss = self.crf_layer(logits, label_ids)
            loss = self.crf_layer(emissions=logits,
                                  tags=label_ids,
                                  mask=torch.tensor(attention_mask, dtype=torch.uint8))
            active_logits = None
            logits = None if self.training else self.crf_layer.decode(emissions=logits, mask=None)
        else:
            loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss
