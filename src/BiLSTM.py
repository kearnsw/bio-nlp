import torch
from torch import nn
import torch.nn.functional as func
from torch.autograd import Variable
from preprocess import load_training_data
from utils import idx_words, text2seq, idx_tags, tags2idx, load_embeddings, to_one_hot
from tqdm import *
import numpy as np
import sys
from argparse import ArgumentParser

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, tag2idx, word2idx, embedding_matrix, hidden_units):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.vocab_size = len(self.word2idx) + 1             # Plus one to include <UNK>
        self.emb_dims = self.embedding_matrix.shape[1]
        self.num_classes = len(tag2idx)
        self.hidden_units = hidden_units

        # Model
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dims)
        self.lstm = nn.LSTM(self.emb_dims, hidden_units)
        self.hidden2tag = nn.Linear(hidden_units, self.num_classes)
        self.softmax = nn.LogSoftmax()

        # Initialize weights
        self.embeddings.weight = nn.Parameter(torch.from_numpy(self.embedding_matrix).float(),
                                              requires_grad=True)      # Don't update weights of embeddings
        self.hidden = self.xavier_initialization()

    def xavier_initialization(self):
        """
        Initializes the hidden units with 
        :return: 
        """
        return (Variable(torch.randn(1, 1, self.hidden_units) * np.sqrt(2 / self.hidden_units)),
                Variable(torch.randn(1, 1, self.hidden_units) * np.sqrt(2 / self.hidden_units)))

    def forward(self, sentence):
        emb = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(emb.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = func.log_softmax(tag_space)
        return tag_scores


if __name__ == "__main__":

    cli_parser = ArgumentParser()
    cli_parser.add_argument("-e", "--embedding", type=str, help="Path to the binary embedding file")
    cli_parser.add_argument("-d", "--emb_dim", type=int, help="Embedding dimensions")
    cli_parser.add_argument("--text", type=str, help="Directory with raw text files")
    cli_parser.add_argument("--ann", type=str, help="Directory with ann files in BRAT standoff format")
    cli_parser.add_argument("--epochs", type=int, help="Number of epochs")
    cli_parser.add_argument("--hidden", type=int, help="Number of hidden units in LSTM")
    args = cli_parser.parse_args()

    # Load training data and split into training and dev
    docs, labels = load_training_data(args.text, args.ann)
    split_idx = int(len(docs) * 0.9)
    x_train = docs[:split_idx]
    y_train = labels[:split_idx]
    x_dev = docs[split_idx:]
    y_dev = docs[split_idx:]

    # Load the word vectors and index the vocabulary of the embeddings and annotation tags
    embeddings, word2idx = load_embeddings(args.embedding, args.emb_dim)
    tag2idx = idx_tags(y_train)

    # Convert data into sequence using indices from embeddings
    x_train = [text2seq(sentence, word2idx, pytorch=True) for sentence in x_train]
    y_train = [tags2idx(tag_seq, tag2idx, pytorch=True) for tag_seq in y_train]

    # Create model and set loss and optimization parameters
    model = BiLSTM(tag2idx, word2idx, embeddings, args.hidden)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1,
                                 weight_decay=0.001)

    print("Number of training examples:{0}".format(len(y_train)))

    # Train
    for epoch in range(args.epochs):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for sentence, labels in tqdm(zip(x_train, y_train), total=len(x_train)):
            model.zero_grad()               # Zero out the gradient from last batch
            pred_tags = model(sentence)
            loss = loss_func(pred_tags, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

    torch.save(model, "model.pkl")
    torch.save(model.state_dict(), "state.pkl")
