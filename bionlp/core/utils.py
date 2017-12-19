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

START_TAG = "<S>"
STOP_TAG = "</S>"


def idx_words(text):
    """
    Index the words of a collection
    :param text: full text of the collection
    :return: a dictionary with words as keys and their indices as values
    """
    word2idx = {}
    for sentence in text:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx


def text2seq(sentence, word2idx, autograd=True):
    """
    Convert a sequence of words into a sequence of indices
    :param sentence: the sentence to be converted to indexed sequence
    :param word2idx: a dictionary with words as keys and their index as the values 
    :param autograd: if true return an autograd variable, else return a numpy array
    :return: 
    """
    sequence = np.zeros(len(sentence), dtype=int)   #LongTensor requires dtype int, e.g. breaks with np.int8
    oov_count = 0

    for i, word in enumerate(sentence):
        # If the word is in the vocabulary then return the index
        # Else assign an index of unknown tag <UNK>, i.e. vocab size + 1
        if word in word2idx:
            sequence[i] = word2idx[word]
        else:
            oov_count += 1
            sequence[i] = len(word2idx)
    
    sys.stdout.write("{0} words were indexed as <UNK>.".format(oov_count))
    sys.stdout.flush()

    # Return embeddings as a np.array or autograd variable
    if autograd:
        return Variable(torch.LongTensor(sequence))
    else:
        return sequence


def tags2idx(tags, tag2idx, autograd=True):
    """
    Convert a sequence of tags to a sequence of indices
    :param tags: list of tags to index
    :param tag2idx: a dictionary with tags as keys, e.g. B-ADR, and their indices as values
    :param autograd: return a pytorch autograd variable or a numpy array 
    :return: 
    """
    indices = [tag2idx[t] for t in tags]
    if autograd:
        return Variable(torch.LongTensor(indices))
    else:
        return np.array(indices)


def idx_tags(tags):
    """
    Index the set of all tags/classes of a collection
    :param tags: the collection of all tags
    :return: a dictionary with unique tags as keys and their index as the value
    """
    tag2idx = {START_TAG: 0, STOP_TAG: 1}
    for seq in tags:
        for tag in seq:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return tag2idx


def load_emb(filename):

    if ".bin" in filename:
        return KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')

    elif ".txt" in filename:
        return KeyedVectors.load_word2vec_format(filename, binary=False, unicode_errors='ignore')


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
    """
    Load the embeddings from a binary or text file and return an embedding matrix
    and an index of the words that have embeddings
    :param filename: a .bin file for Word2Vec format embeddings or .txt for GloVe embeddings
    :param dims: number of dimensions of the embeddings
    :return: a tuple of (embedding matrix, word2idx)
    """
    return generate_emb_matrix(load_emb(filename), dims)


def to_one_hot(idx, nb_classes):
    """
    Converts a single index to a one-hot representation
    e.g. 3 -> [0,0,0,1,0] if there are 5 classes 
    :param idx: the index of the class starting from 0
    :param nb_classes: total number of classes
    :return: a one-hot vector representation of the class
    """
    one_hot = np.zeros(1, nb_classes)
    one_hot[idx] = 1
    return one_hot


def seq_to_one_hot(array, nb_classes):
    """
    Convert a sequence of indexed classes, e.g. w/ 4 classes [3, 2, 0]
    into a one-hot vector representation [[0,0,1,0][0,1,0,0][1,0,0,0]]
    :param array: a flat array of class indices
    :param nb_classes: number of classes total
    :return: a one-hot vector representation with dims (array_len, nb_classes)
    """
    one_hot = np.zeros(len(array), nb_classes)
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
    """
    Segments a list into batches of specified size (last batch may be < batch_size)
    :param data: the list to be separated into batches 
    :param batch_size: the number of samples in each batch
    :return: a list of lists with dimensions (num_batches, batch_size)
    """
    num_batches = int(len(data)/batch_size)
    batches = np.empty(num_batches, dtype=list)
    for k in range(num_batches - 1):
        batches[k] = data[k*batch_size:(k+1)*batch_size]

    # Add the remaining samples to the last batch
    batches[num_batches - 1] = data[-batch_size:]

    return batches


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
