import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from bionlp.core.preprocess import load_data
from bionlp.core.utils import idx_words, text2seq, idx_tags, tags2idx, load_embeddings, to_one_hot, plot_loss
from bionlp.core.utils import create_batches
from tqdm import *
import numpy as np
import sys
import os
from argparse import ArgumentParser

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, tag2idx, word2idx, embedding_matrix, hidden_units, num_layers):
        super().__init__()

        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.vocab_size = len(self.word2idx) + 1  # Plus one to include <UNK>
        self.emb_dims = self.embedding_matrix.shape[1]
        self.num_classes = len(tag2idx)
        self.hidden_units = hidden_units
        self.nb_layers = num_layers
        self.bidirectional = True

        # Model
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dims)
        self.lstm = nn.LSTM(input_size=self.emb_dims,
                            hidden_size=hidden_units // 2,
                            num_layers= self.nb_layers,
                            bidirectional=self.bidirectional)
        self.hidden2tag = nn.Linear(hidden_units, self.num_classes)
        self.softmax = nn.LogSoftmax()

        # Initialize weights
        self.embeddings.weight = nn.Parameter(torch.from_numpy(self.embedding_matrix).float(),
                                              requires_grad=True)  # Don't update weights of embeddings
        self.hidden, self.cell_state = self.xavier_initialization()

    def xavier_initialization(self):
        """
        Initializes the hidden unit and cell state of the LSTM 
        :return: a matrix of initial weights as an Autograd variable
        """

        if self.bidirectional:
            return (Variable(torch.randn(self.nb_layers * 2, 1, self.hidden_units // 2) * np.sqrt(2 / self.hidden_units)),
                    Variable(torch.randn(self.nb_layers * 2, 1, self.hidden_units // 2) * np.sqrt(2 / self.hidden_units)))
        else:
            return (Variable(torch.randn(self.nb_layers, 1, self.hidden_units) * np.sqrt(2 / self.hidden_units)),
                    Variable(torch.randn(self.nb_layers, 1, self.hidden_units) * np.sqrt(2 / self.hidden_units)))

    def forward(self, sentence):
        emb = self.embeddings(sentence)
        lstm_out, (self.hidden, self.cell_state) = self.lstm(emb.view(len(sentence), 1, -1), (self.hidden, self.cell_state))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), self.hidden_units))
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
    cli_parser.add_argument("--batch-size", type=int, default=10)
    cli_parser.add_argument("--layers", type=int, default=1, help="number of layers")
    cli_parser.add_argument("--cuda", action="store_true")
    cli_parser.add_argument("--models", type=str, help="model directory")
    args = cli_parser.parse_args()

    state_file = os.sep.join([args.models, "hid{0}_lay{1}_emb{2}.states".format(args.hidden, args.layers, args.emb_dim)])
    model_file = os.sep.join([args.models, "hid{0}_lay{1}_emb{2}.model".format(args.hidden, args.layers, args.emb_dim)])

    # Load training data and split into training and dev
    docs, labels = load_data(args.text, args.ann, shuffle=True)
    split_idx = int(len(docs) * 0.90)

    x_train = docs[:split_idx]
    y_train = labels[:split_idx]
    x_dev = docs[split_idx:]
    y_dev = labels[split_idx:]

    # Load the word vectors and index the vocabulary of the embeddings and annotation tags
    embeddings, word2idx = load_embeddings(args.embedding, args.emb_dim)
    tag2idx = idx_tags(y_train)

    # Convert data into sequence using indices from embeddings
    x_train = [text2seq(sentence, word2idx) for sentence in x_train]
    y_train = [tags2idx(tag_seq, tag2idx) for tag_seq in y_train]
    x_dev = [text2seq(sentence, word2idx) for sentence in x_dev]
    y_dev = [tags2idx(tag_seq, tag2idx) for tag_seq in y_dev]

    # Create model and set loss and optimization parameters
    model = BiLSTM(tag2idx, word2idx, embeddings, args.hidden, args.layers)
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1,
                                 weight_decay=0.01)

    # Load checkpoint file if it exists
    if os.path.isfile(state_file):
        print("Initializing model state from file...")
        model.load_state_dict(torch.load(state_file))

    # Store loss from each epoch for early stopping and plotting loss-curve
    loss = np.zeros(args.epochs)

    # Move model parameters and data to GPU
    if args.cuda:
        model.cuda()
        x_train = [var.cuda() for var in x_train]
        y_train = [var.cuda() for var in y_train]

    # Create batches
    mini_batches = create_batches(list(zip(x_train, y_train)), 100)

    print("Number of training examples:{0}".format(len(y_train)))
    # Train model using mini-batches
    for epoch in range(args.epochs):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for mini_batch in tqdm(mini_batches, total=len(mini_batches)):
            batch_loss = Variable(torch.FloatTensor([0]))  # Zero out the loss from last batch
            model.zero_grad()                              # Zero out the gradient from last batch
            for doc, tags in mini_batch:
                model.hidden, model.cell_state = model.xavier_initialization()
                pred_tags = model(doc)
                batch_loss += loss_func(pred_tags, tags)

            # Backpropagate the loss for each mini-batch
            batch_loss.backward()
            optimizer.step()
            loss[epoch] += batch_loss.data[0]

        sys.stdout.write("Loss: {0}\n".format(loss[epoch]/len(x_train)))
        sys.stdout.flush()

        # Early Stopping
        if epoch > 0 and loss[epoch - 1] - loss[epoch] <= 0.01:
            break

        # Checkpoint
        torch.save(model.state_dict(), state_file)

    # Save best model
    torch.save(model, model_file)

    plot_loss(loss)
