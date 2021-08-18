# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 7/15/21

import torch
from torch import nn


class CharLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CharLSTM, self).__init__()
        self.lstm_encoder = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

    def forward(self):
        pass


class WordLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WordLSTM, self).__init__()
        self.lstm_encoder = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

    def forward(self):
        pass


class LinearCRF(nn.Module):
    def __init__(self):
        super(LinearCRF, self).__init__()

    def forward(self):
        pass


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self):
        pass


class FeatureEmbeddings(nn.Module):
    def __init__(self):
        super(FeatureEmbeddings, self).__init__()

    def forward(self):
        pass


class BiLSTM_CRF(nn.Module):
    def __init__(self):
        super(BiLSTM_CRF, self).__init__()

    def forward(self):
        pass