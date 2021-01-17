#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
MAX_LENGTH = 16


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        embedded = input
        hidden = hidden.transpose(0, 1)
        embedded = embedded.permute(1, 0, 2)
        attn_weights = F.softmax(self.attn(hidden.squeeze(1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden.transpose(0, 1))
        output = self.out(output.squeeze(1)).unsqueeze(1)
        return output, hidden, attn_weights

    def initHidden(self, batchsize):
        return torch.zeros(1, batchsize, self.hidden_size)

class EncoderDecoderAtt(nn.Module):
    def __init__(self, encoder, decoder, time_step, **kwargs):
        super(EncoderDecoderAtt, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.step = time_step
    def forward(self, X, *args):
        encoder_outputs, hidden = self.encoder(X)
        print('encoder_outputs: ', encoder_outputs.shape)
        attn_weights = []
        outputs = []
        for i in range(self.step):
            output, hidden, attn_weight = self.decoder(hidden, hidden, encoder_outputs)
            attn_weights.append(attn_weight)
            if i == 0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), 1)
        return outputs, attn_weights