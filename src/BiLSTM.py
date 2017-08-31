import torch
from torch import nn
import torch.nn.functional as func
from torch.autograd import Variable
from src.preprocess import load_training_data
from src.utils import idx_words, text2seq, idx_tags, tags2idx, load_embeddings, to_one_hot
import numpy as np
import sys
from argparse import ArgumentParser

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.view(-1).data.tolist()[0]


class BiLSTM(nn.Module):

    def __init__(self, tag2idx, hidden_dims):
        super().__init__()
        self.embedding_matrix, self.word2idx = load_embeddings(args.embedding, args.emb_dim)
        self.vocab_size = len(word2idx) + 1             # Plus one to include <UNK>
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

    data = load_training_data(args.text, args.ann)
    training_data = data[:10]
    word2idx = idx_words(training_data)
    tag2idx = idx_tags(training_data)
    model = BiLSTM(tag2idx, 64)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1,
                                 weight_decay=0.001)

    for epoch in range(20):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for sentence, tags in training_data:
            model.zero_grad()

            idx_seq = text2seq(sentence, word2idx)

            true_tags = Variable(torch.LongTensor([to_one_hot(tag2idx[t], model.num_tags) for t in tags]))
            pred_tags = model(idx_seq)
            print(true_tags)
            print(pred_tags)
            loss = loss_func(pred_tags, true_tags)
            loss.backward()
            optimizer.step()

    torch.save(model, "model.pkl")
    torch.save(model.state_dict(), "state.pkl")
    prediction = model(text2seq(data[200][0], word2idx))
    truth = tags2idx(data[200][1], tag2idx)
    print(truth.size())
    truth = truth.resize(1, truth.size()[0])
    print(truth)
    print(prediction)
    print(tag2idx)
