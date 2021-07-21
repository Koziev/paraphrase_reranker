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

from synonymy_detector import SynonymyDetector


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

    def check_pair(self, phrase1, phrase2):
        tokenized_pair = []
        for sent in [phrase1, phrase2]:
            tx = self.bert_tokenizer.encode(sent)
            tx = tx + [self.pad_token_id] * (self.max_len - len(tx))
            t = torch.tensor(tx)
            tokenized_pair.append(t)

        y = self.model(tokenized_pair[0].unsqueeze(0).to(device), tokenized_pair[1].unsqueeze(0).to(device))[0].item()
        return y


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
