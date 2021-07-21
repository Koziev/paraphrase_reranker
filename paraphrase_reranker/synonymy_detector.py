import pandas as pd
import numpy as np
import random
import os
import pickle

import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformers


class SynonymyDetector(nn.Module):
    def __init__(self, computed_params):
        super(SynonymyDetector, self).__init__()
        self.bert_model = computed_params['bert_model']

        sent_emb_size = computed_params['sent_emb_size']

        self.arch = 2

        if self.arch == 1:
            self.fc1 = nn.Linear(sent_emb_size*2, 20)
            self.fc2 = nn.Linear(20, 1)
        elif self.arch == 2:
            embedding_dim = computed_params['sent_emb_size']
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
            self.fc1 = nn.Linear(in_features=embedding_dim*4, out_features=20)
            self.fc2 = nn.Linear(in_features=20, out_features=1)
        else:
            raise NotImplementedError()

    def forward(self, x1, x2):
        #with torch.no_grad():

        if self.arch == 1:
            z1 = self.bert_model(x1)[0].sum(dim=-2)
            z2 = self.bert_model(x2)[0].sum(dim=-2)

            #merged = torch.cat((z1, z2, torch.abs(z1 - z2)), dim=-1)
            #merged = torch.cat((z1, z2, torch.abs(z1 - z2), z1 * z2), dim=-1)
            merged = torch.cat((z1, z2), dim=-1)

            merged = self.fc1(merged)

            #merged = torch.relu(merged)
            #output = torch.sigmoid(merged)

            merged = torch.relu(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 2:
            z1 = self.bert_model(x1)[0]
            z2 = self.bert_model(x2)[0]

            out1, (hidden1, cell1) = self.rnn(z1)
            v1 = out1[:, -1, :]

            out2, (hidden2, cell2) = self.rnn(z2)
            v2 = out2[:, -1, :]

            v_sub = torch.sub(v1, v2)
            v_mul = torch.mul(v1, v2)

            merged = torch.cat((v1, v2, v_sub, v_mul), dim=-1)

            merged = self.fc1(merged)
            merged = torch.relu(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        else:
            raise NotImplementedError()

        return output
