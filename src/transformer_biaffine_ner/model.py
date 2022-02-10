# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

import torch
from torch import nn
from transformer_ner.model_utils import _calculate_loss


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, hidden_dim=0, num_hidden_layers=0):
        super().__init__()
        self.weight = None
        # TODO: test Relu and LeakyRelu (negative_slope=0.1) linear activation
        # TODO: test if dropout need (SharedDropout)
        activation_fct = activation if activation else nn.GELU()
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
            self.weight = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim), activation_fct)
        else:
            # only one linear layer
            self.weight = nn.Sequential(nn.Linear(input_dim, output_dim), activation_fct)

    def forward(self, x):
        return self.weight(x)


class Biaffine(nn.Module):
    def __init__(self, input_dim, output_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.bx = bias_x
        self.by = bias_y
        self.U = torch.nn.Parameter(
            torch.Tensor(input_dim + int(bias_x), output_dim, input_dim + int(bias_y)))
        # TODO: use normal init; we can test other init method: xavier, kaiming, ones
        nn.init.normal_(self.U)

    def forward(self, x, y):
        # add bias
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
        """
        biaffine_mappings = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)  # einsum known to be slow in some cases

        return biaffine_mappings


class BiaffineLayer(nn.Module):
    """
        ref:
            https://aclanthology.org/2020.acl-main.577.pdf
            https://github.com/geasyheart/biaffine_ner.git
    """
    def __init__(self, config):
        super().__init__()
        # TODO: option to use both bert output last and second last hidden states
        # TODO: add flag for different MLP activation function
        # mlp_input_dim = config.hidden_size if config.include_only_bert_last_hidden else config.hidden_size*2
        mlp_input_dim = config.hidden_size
        mlp_output_dim = config.mlp_dim if config.mlp_dim > 0 else config.hidden_size
        self.ffnns = MLP(mlp_input_dim, mlp_output_dim)  # ffnns: feed forward neural network start
        self.ffnne = MLP(mlp_input_dim, mlp_output_dim)  # ffnne: feed forward neural network end
        self.biaffine = Biaffine(mlp_output_dim, config.num_labels)
        self.num_labels = config.num_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, attention_mask=None, label_ids=None):
        s_logits = self.ffnns(x)
        e_logits = self.ffnne(x)
        logits = self.biaffine(s_logits, e_logits)

        loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss


class TransfomerBiaffineNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_biaffine = config.use_biaffine if hasattr(config, "use_biaffine") else None
        self.biaffine = BiaffineLayer(config) if self.use_biaffine else None

    def forward(self, input_ids, attention_mask, label_ids):
        sequence_output = None
        if self.use_biaffine:
            logits, active_logits, loss = self.biaffine(sequence_output, attention_mask, label_ids)

