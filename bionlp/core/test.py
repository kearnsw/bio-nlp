#!/usr/bin/env python

import torch
from bionlp.core.BiLSTM import BiLSTM
from bionlp.core.preprocess import strip_punctuation, load_data
from bionlp.core.utils import text2seq, idx_words, idx_tags, tags2idx

fn = "model.pkl"
print("Loading Model...")
model = torch.load(fn)


s = strip_punctuation("I have a severe pain sore throat and sometimes lipitor ativan allegra and stuff make me moody").split()
sequence = text2seq(s, model.word2idx)
data = load_data()
prediction = model(sequence)
matrix = prediction.data.numpy()
idx2tag = {v:k for k,v in model.tag2idx.items()}
tagged_sequence = ["({0}/{1})".format(s[i], idx2tag[row.argmax()]) for i,row in enumerate(matrix)]

print(" ".join(tagged_sequence))

