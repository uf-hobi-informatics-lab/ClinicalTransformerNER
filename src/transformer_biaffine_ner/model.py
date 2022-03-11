# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

import torch
from torch import nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformer_ner.model_utils import FocalLoss


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=0, num_hidden_layers=0):
        super().__init__()
        self.weight = None
        # TODO: test Relu and LeakyRelu (negative_slope=0.1) linear activation
        activation_fct = nn.GELU()
        # we hard code dropout as 0.1
        dropout = nn.Dropout(0.1)
        if num_hidden_layers and hidden_dim:
            # if num_hidden_layers = 1, then we have two layers
            layers = []
            for i in range(num_hidden_layers):
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                # should test Relu and LeakyRelu (negative_slope=0.1)
                layers.append(activation_fct)
            self.weight = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim), activation_fct, dropout)
        else:
            # only one linear layer
            self.weight = nn.Sequential(nn.Linear(input_dim, output_dim), activation_fct, dropout)

    def forward(self, x):
        return self.weight(x)


class Biaffine(nn.Module):
    def __init__(self, input_dim, output_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.bx = bias_x
        self.by = bias_y

        # biaffine metrics
        self.U = nn.Parameter(
            torch.Tensor(input_dim + int(bias_x), output_dim, input_dim + int(bias_y)))

        # use xavier_norm init; we can test other init method: norm_, kaiming, ones
        nn.init.xavier_normal_(self.U)

    def forward(self, x, y):
        # add bias => Wm(hs(i)⊕he(i)) + bm
        if self.bx:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.by:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)

        """
        t1: [b, s, v]
        t2: [b, s, v]
        U: [v, o, v]

        m = t1*U => [b,s,o,v] => [b, s*o, v]
        m*t2.T => [b, s*o, v] * [b, v, s] => [b, s, o, s] => [b, s, s, o]: this is the mapping table
        
        from https://aclanthology.org/2020.acl-main.577.pdf
        hs(i) = FFNNs(xsi)
        he(i) = FFNNe(xei)
        we implement following
        rm(i) = hs(i)T*U*he(i) + (Wm(hs(i)⊕he(i)) + bm)
        """

        biaffine_rep = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)

        return biaffine_rep


class _Biaffine(nn.Module):
    # implementation without torch.einsum
    def __init__(self, input_dim, output_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.bx = bias_x
        self.by = bias_y
        self.output_dim = output_dim

        linear_input_dim = input_dim + int(bias_x)
        new_output_dim = input_dim + int(bias_y)
        linear_output_dim = output_dim * new_output_dim
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, x, y):
        assert x.shape == y.shape
        bz, seq_len, _ = x.shape

        if self.bx:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.by:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)

        affine = self.linear(x)
        affine = torch.reshape(affine, (bz, seq_len*self.output_dim, -1))
        biaffine = torch.matmul(affine, torch.permute(y, (0, 2, 1)))
        biaffine = torch.permute(biaffine, (0, 2, 1))
        biaffine = torch.reshape(biaffine, (bz, seq_len, seq_len, -1))
        # squeeze last dim if 1
        biaffine = torch.squeeze(biaffine, dim=-1)

        return biaffine


class BiaffineLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: test if use bert output last and second last hidden states are better than just last hidden states
        # TODO: add flag for different MLP activation function
        # mlp_input_dim = config.hidden_size if config.include_only_bert_last_hidden else config.hidden_size*2
        mlp_input_dim = config.hidden_size
        mlp_output_dim = config.mlp_dim if config.mlp_dim > 0 else config.hidden_size
        mlp_hidden_dim = config.mlp_hidden_dim
        # ffnns: feed forward neural network start
        self.ffnns = MLP(mlp_input_dim, mlp_output_dim, mlp_hidden_dim, config.mlp_layers)
        # ffnne: feed forward neural network end
        self.ffnne = MLP(mlp_input_dim, mlp_output_dim, mlp_hidden_dim, config.mlp_layers)
        self.biaffine = Biaffine(mlp_output_dim, config.num_labels)

        self.config = config
        if hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_loss_gamma)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def biaffine_loss_calculation(self, preds, labels, masks):
        active_idx = masks.view(-1) == 1

        preds = preds.reshape(-1, self.config.num_labels)
        preds_masked = preds[active_idx]

        labels_masked = labels.view(-1)[active_idx]

        loss = self.loss_fct(preds_masked, labels_masked)

        return loss

    def forward(self, x, labels=None, loss_mask=None):
        s_logits = self.ffnns(x)
        e_logits = self.ffnne(x)
        logits = self.biaffine(s_logits, e_logits)

        loss = self.biaffine_loss_calculation(logits, labels, loss_mask)

        return logits, loss


class TransformerBiaffineNerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.init_in_training:
            self.lm = AutoModel.from_pretrained(config.base_model_path, config=config)
        else:
            self.lm = AutoModel.from_config(config=config)

        self.biaffine = BiaffineLayer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def resize_token_embeddings(self, new_size):
        self.lm.resize_token_embeddings(new_size)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                masks=None):
        # obtain representations from pretrained language model
        lm_representatons = self.lm(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids)
        # here we only get the last layers; can get last two layers and do avg
        tokens_representations = lm_representatons[0]
        tokens_representations = self.dropout(tokens_representations)

        logits, loss = self.biaffine(tokens_representations, labels, masks)
        return logits, loss
