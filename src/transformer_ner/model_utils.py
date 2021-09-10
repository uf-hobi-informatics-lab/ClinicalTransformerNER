import torch
from torch import nn
from torch.nn import functional as F
import math


def xavier_init(*layers):
    for layer in layers:
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias.data)


def kaiming_init(*layers):
    for layer in layers:
        nn.init.kaiming_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias.data)


def gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, inputs=None, local_ctx=None):
        mask, dropout = get_mask(inputs, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return inputs.masked_fill(mask, 0) * ctx.scale
        else:
            return inputs

    @staticmethod
    def backward(ctx, grad_output=None):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class SharedDropout(nn.Module):
    """
        SharedDropout differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.
        Args:
            p (float):
                The probability of an element to be zeroed. Default: 0.5.
            batch_first (bool):
                If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
                Default: ``True``.
        Examples:
            >>> x = torch.ones(1, 3, 5)
            >>> nn.Dropout()(x)
            tensor([[[0., 2., 2., 0., 0.],
                     [2., 2., 0., 2., 2.],
                     [2., 2., 2., 2., 0.]]])
            >>> SharedDropout()(x)
            tensor([[[2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.]]])
        """

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        """
        :param x:
        :return:
        """

        if not self.training:
            return x
        if self.batch_first:
            mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
        else:
            mask = self.get_mask(x[0], self.p)
        x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = gelu(pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def _calculate_loss(logits, attention_mask, label_ids, loss_fct=None, num_labels=2):
    if attention_mask is not None:
        active_idx = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, num_labels)[active_idx]
        active_labels = label_ids.view(-1)[active_idx]
    else:
        active_logits = logits.view(-1, num_labels)
        active_labels = label_ids.view(-1)

    loss = loss_fct(active_logits, active_labels)

    return loss, active_logits