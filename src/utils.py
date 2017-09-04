"""
Utilities for NN models using Pytorch
@author: Will Kearns
@license: GPL 3.0
"""

import torch
from torch.autograd import Variable
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from tqdm import *
import matplotlib.pyplot as plt
import seaborn as sb

START_TAG = "<s>"
STOP_TAG = "</s>"

def idx_words(text):
    word2idx = {}
    for sentence in text:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx


def text2seq(sentence, word2idx, pytorch=False):
    sequence = np.zeros(len(sentence), dtype=int)   #LongTensor requires dtype int, e.g. breaks with np.int8
    for i, word in enumerate(sentence):
        if word in word2idx:
            sequence[i] = word2idx[word]
        else:
            sequence[i] = len(word2idx)

    if pytorch:
        tensor = torch.LongTensor(sequence)
        return Variable(tensor)
    else:
        return sequence


def tags2idx(seq, tag2idx, pytorch=False):
    seq = [tag2idx[t] for t in seq]
    if pytorch:
        tensor = torch.LongTensor(seq)
        return Variable(tensor)
    else:
        return np.array(seq)


def idx_tags(tags):
    tag2idx = {START_TAG: 0, STOP_TAG: 1}
    for seq in tags:
        for tag in seq:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return tag2idx


def load_emb(filename):
    if ".bin" in filename:
        return KeyedVectors.load_word2vec_format(filename, binary=True, limit=100)

    elif ".txt" in filename:
        word2vec = {}
        print("Loading Word Embeddings...")
        with open(filename, "r") as f:
            lines = f.read().split("\n")
            for line in tqdm(lines):
                if line:
                    cols = line.split()
                    word = cols[0]
                    embedding = np.array(cols[1:], dtype=np.float)
                    word2vec[word] = embedding
        return word2vec


def generate_emb_matrix(word2vec, dims):
    word2idx = {}
    print("Initializing Embedding Matrix...")

    if type(word2vec) == dict:
        embedding_matrix = np.zeros((len(word2vec) + 1, dims))
        for idx, word in enumerate(tqdm(word2vec)):
            embedding_matrix[idx] = word2vec[word]
            word2idx[word] = idx

    else:
        embedding_matrix = np.zeros((len(word2vec.vocab) + 1, dims))
        for idx, word in enumerate(tqdm(word2vec.vocab)):
            embedding_matrix[idx] = word2vec.word_vec(word)
            word2idx[word] = idx

    embedding_matrix[-1] = np.random.rand(1, dims)          # Add vector for <UNK>

    return embedding_matrix, word2idx


def load_embeddings(filename, dims):
    return generate_emb_matrix(load_emb(filename), dims)


def to_one_hot(idx, shape):
    one_hot = np.zeros(shape)
    one_hot[idx] = 1
    return one_hot


def seq_to_one_hot(array, shape):
    one_hot = np.zeros(shape)
    for idx, value in enumerate(array):
        if not value:
            return np.array([])
        one_hot[idx][value] = 1
    return one_hot


def handle_unk_words(word):
    # dosage word
    if "mg" in word:
        return

    # time
    if any(["year", "weeks", "wks", "days"]) in word:
        return

    # misspelled medical terms
    if "olest" in word:
        return "cholesterol"


def create_batches(data, batch_size):
    num_batches = int(len(data)/batch_size)
    batch = np.empty(num_batches, dtype=list)
    for k in range(num_batches - 1):
        batch[k] = data[k*batch_size:(k+1)*batch_size]
    batch[num_batches - 1] = data[-batch_size:]

    return batch


def plot_loss(loss_array):
    # Use seaborn style
    sb.set_style("darkgrid")

    # Plot data and label axes
    plt.plot(loss_array)
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.title("Loss over time")

    # Save to disk
    plt.savefig("loss_plot.png")

if __name__ == "__main__":

    emb_matrix, word2idx = generate_emb_matrix(load_emb("../vectors/PubMed-shuffle-win-30.bin"), 200)

    nausea = emb_matrix[word2idx["nausea"]]
    vomitting = emb_matrix[word2idx["vomitting"]]
    diabetes = emb_matrix[word2idx["diabetes"]]
    the = emb_matrix[word2idx["the"]]

    print(np.dot(nausea, vomitting.T))
    print(np.dot(nausea, diabetes.T))
    print(np.dot(nausea, the.T))
