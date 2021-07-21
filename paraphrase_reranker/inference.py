import pandas as pd
import numpy as np
import random
import pathlib
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

from .synonymy_detector import SynonymyDetector


class ParaphraseRanker:
    def __init__(self, device='cpu'):
        self.device = device

    def load(self, model_dir):
        if model_dir is None:
            model_dir = str(pathlib.Path(__file__).resolve().parent)

        weights_path = os.path.join(model_dir, "synonymy_model.pt")
        config_path = os.path.join(model_dir, 'paraphrase_detector_6.cfg')

        with open(config_path, 'rb') as f:
            self.cfg = pickle.load(f)

        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(self.cfg['bert_model_name'], do_lower_case=False)
        self.bert_model = transformers.BertModel.from_pretrained(self.cfg['bert_model_name'])

        self.bert_model.to(device)
        self.bert_model.eval()

        computed_params = {'sent_emb_size': self.cfg['sent_emb_size'],
                           'bert_tokenizer': self.bert_tokenizer,
                           'bert_model': self.bert_model}

        self.max_len = self.cfg['max_len']
        self.pad_token_id = self.bert_tokenizer.pad_token_id

        self.model = SynonymyDetector(computed_params).to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))

    def tokenize(self, text):
        tx = self.bert_tokenizer.encode(text)
        l = len(tx)
        if l < self.max_len:
            tx = tx + [self.pad_token_id] * (self.max_len - l)
        elif l > self.max_len:
            tx = tx[:self.max_len]

        return tx

    def check_pair(self, phrase1, phrase2):
        tokenized_pair = []
        for sent in [phrase1, phrase2]:
            tokenized_pair.append(torch.tensor(self.tokenize(sent)))

        y = self.model(torch.tensor(tokenized_pair[0]).unsqueeze(0).to(device),
                       torch.tensor(tokenized_pair[1]).unsqueeze(0).to(device))[0].item()
        return y

    def rerank(self, phrase1, paraphrases):
        n = len(paraphrases)
        x1 = np.zeros((n, self.max_len), dtype=np.int32)
        x2 = np.zeros((n, self.max_len), dtype=np.int32)

        tokenized_phrase1 = self.tokenize(phrase1)
        for i, phrase2 in enumerate(paraphrases):
            x1[i, :] = tokenized_phrase1
            x2[i, :] = self.tokenize(phrase2)

        ys = self.model(torch.tensor(x1).to(device), torch.tensor(x2).to(device)).numpy()
        paraphrases2 = sorted(zip(paraphrases, ys), key=lambda z: -z[1])
        return paraphrases2


if __name__ == '__main__':
    model_dir = '../tmp'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ranker = ParaphraseRanker(device)
    ranker.load(model_dir)

    pairs = [('кошка ловит мышку', 'кошка охотится на мышку'),
             ('меня зовут илья', 'мое имя илья')]
    for pair in pairs:
        y = ranker.check_pair(pair[0], pair[1])
        print('{} <==> {} ==> {}'.format(pair[0], pair[1], y))

    px2 = ranker.rerank('кошка ловит мышку', ['кошка охотится на мышку', 'мыши едят зерно', 'у кошки зеленые глаза'])
    print(px2)
