import torch
from torch import nn
import torch.nn.functional as func
from torch.autograd import Variable
from preprocess import load_training_data
from utils import idx_words, text2seq, idx_tags, tags2idx, load_embeddings, to_one_hot
import numpy as np
import sys
from argparse import ArgumentParser

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, tag2idx, hidden_dims):
        super().__init__()
        self.embedding_matrix, self.word2idx = load_embeddings(args.embedding, args.emb_dim)
        self.tag2idx = tags2idx
        self.vocab_size = len(self.word2idx) + 1             # Plus one to include <UNK>
        self.emb_dims = self.embedding_matrix.shape[1]
        self.num_tags = len(tag2idx)
        self.hidden_dims = hidden_dims

        # Model
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dims)
        self.lstm = nn.LSTM(self.emb_dims, hidden_dims)
        self.hidden2tag = nn.Linear(hidden_dims, self.num_tags)

        # Initialize weights
        self.embeddings.weight = nn.Parameter(torch.from_numpy(self.embedding_matrix).float(),
                                              requires_grad=True)      # Don't update weights of embeddings
        self.hidden = self.xavier_initialization()

    def xavier_initialization(self):
        return (Variable(torch.randn(1, 1, self.hidden_dims) * np.sqrt(2/self.hidden_dims)),
                Variable(torch.randn(1, 1, self.hidden_dims) * np.sqrt(2/self.hidden_dims)))

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = func.log_softmax(tag_space)
        return tag_scores


if __name__ == "__main__":
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-e", "--embedding", type=str, help="Path to the binary embedding file")
    cli_parser.add_argument("-d", "--emb_dim", type=int, help="Embedding dimensions")
    cli_parser.add_argument("--text", type=str, help="Directory with raw text files")
    cli_parser.add_argument("--ann", type=str, help="Directory with ann files in BRAT standoff format")
    args = cli_parser.parse_args()

    # Load training data and split into training and dev
    docs, labels = load_training_data(args.text, args.ann)
    split_idx = 10
    x_train = docs[:split_idx]
    y_train = labels[:split_idx]
    x_dev = docs[split_idx:]
    y_dev = docs[split_idx:]

    # Create the model
    tag2idx = idx_tags(y_train)
    model = BiLSTM(tag2idx, 64)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1,
                                 weight_decay=0.001)

    # Convert data into sequence using indices from embeddings
    x_train = [text2seq(sentence, model.word2idx, pytorch=True) for sentence in x_train]
    y_train = [tags2idx(tag_seq, model.tag2idx, pytorch=True) for tag_seq in y_train]

    # Train
    for epoch in range(20):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for sentence, labels in zip(x_train, y_train):
            model.zero_grad()
            true_tags = Variable(torch.LongTensor(labels))
            pred_tags = model(sentence)
            print(true_tags)
            print(pred_tags)
            loss = loss_func(pred_tags, true_tags)
            loss.backward()
            optimizer.step()

    torch.save(model, "model.pkl")
    torch.save(model.state_dict(), "state.pkl")
