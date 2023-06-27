import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
from transformers import BertTokenizerFast, AutoTokenizer

from utils import TrainOptions


opt = TrainOptions().parse()
if opt.model_name == 'BanglaBert':
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/mbert-bengali-ner")

else:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
def load_datasets(path):
    df = pd.read_csv(path)
    return df

def ids_exhge_labels(opt):
    df = load_datasets(opt.dataroot)
    labels = [i.split() for i in df['labels'].values.tolist()]
    unique_labels = set()

    for lb in labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
    return labels_to_ids, ids_to_labels


label_all_tokens = False

def align_label(texts, labels, opt):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=opt.max_token_length, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, opt):

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = opt.max_token_length, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j,opt) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels