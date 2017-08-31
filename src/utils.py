"""
Utilities for NN models using Pytorch
@author: Will Kearns
@license: GPL 3.0
"""

import torch
from torch import autograd
from config import START_TAG, STOP_TAG, EMBEDDING_FILE
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import seaborn as sb

def idx_words(text):
    word2idx = {}
    for sentence, _ in text:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx


def text2seq(sentence, word2idx):
    sequence = np.zeros(len(sentence), dtype=int)   #LongTensor requires dtype int, e.g. breaks with np.int8
    for i, word in enumerate(sentence):
        if word in word2idx:
            sequence[i] = word2idx[word]
            print(sequence[i])
        else:
            sequence[i] = len(word2idx) + 1
    tensor = torch.LongTensor(sequence)
    return autograd.Variable(tensor)


def tags2idx(seq, tag2idx):
    tensor = torch.LongTensor([tag2idx[t] for t in seq])
    return autograd.Variable(tensor)


def idx_tags(data):
    tag2idx = {START_TAG: 0, STOP_TAG: 1}
    for _, seq in data:
        for tag in seq:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return tag2idx


def load_emb(filename):
    return KeyedVectors.load_word2vec_format(filename, binary=True)


def generate_emb_matrix(word2vec, dims):
    word2idx = {}
    embedding_matrix = np.zeros((len(word2vec.vocab) + 1, dims))
    for idx, word in enumerate(word2vec.vocab):
        embedding_matrix[idx] = word2vec.word_vec(word)
        word2idx[word] = idx

    embedding_matrix[-1] = np.random.rand(1, dims)          # Add vector for <UNK>

    return embedding_matrix, word2idx


def load_embeddings(filename, dims):
    return generate_emb_matrix(load_emb(filename), dims)

if __name__ == "__main__":

    emb_matrix, word2idx = generate_emb_matrix(load_emb(EMBEDDING_FILE), 200)

    nausea = emb_matrix[word2idx["nausea"]]
    vomitting = emb_matrix[word2idx["vomitting"]]
    diabetes = emb_matrix[word2idx["diabetes"]]
    the = emb_matrix[word2idx["the"]]

    print(np.dot(nausea, vomitting.T))
    print(np.dot(nausea, diabetes.T))
    print(np.dot(nausea, the.T))
