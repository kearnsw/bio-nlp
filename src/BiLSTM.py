import torch
from torch import nn
import torch.nn.functional as func
from torch.autograd import Variable
from src.preprocess import load_training_data
from src.utils import idx_words, text2seq, idx_tags, tags2idx, load_embeddings
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
        self.embedding_matrix, self.word2idx = load_embeddings(cli_parser.embedding, cli_parser.emb_dim)
        self.vocab_size = len(word2idx) + 1             # Plus one to include <UNK>
        self.emb_dims = self.embedding_matrix.shape[1]
        self.num_tags = len(tag2idx)
        self.hidden_dims = hidden_dims

        # Model
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dims)
        self.lstm = nn.LSTM(self.emb_dims, hidden_dims)
        self.hidden2tag = nn.Linear(hidden_dims, 1)
        self.hidden = self.init_hidden()

        # Set weights
        self.embeddings.weight = nn.Parameter(torch.from_numpy(self.embedding_matrix).float(),
                                              requires_grad=True)      # Don't update weights of embeddings

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_dims)),
                Variable(torch.zeros(1, 1, self.hidden_dims)))

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return tag


if __name__ == "__main__":
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-e", "--embedding", type=str, help="Path to the binary embedding file")
    cli_parser.add_argument("-d", "--emb_dim", type=int, help="Embedding dimensions")
    cli_parser.add_argument("--text-dir", type=str, help="Directory with raw text files")
    cli_parser.add_argument("--annotation-dir", type=str, help="Directory with ann files in BRAT standoff format")
    args = cli_parser.parse_args()

    data = load_training_data(args["text-dir"], args["annotation-dir"])
    training_data = data[:10]
    word2idx = idx_words(training_data)
    tag2idx = idx_tags(training_data)
    model = BiLSTM(tag2idx, 64)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1,
                                 weight_decay=0.001)

    for epoch in range(20):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for sentence, tags in training_data:
            model.zero_grad()

            idx_seq = text2seq(sentence, word2idx)

            true_tags = Variable(torch.LongTensor([tag2idx[t] for t in tags]))
            pred_tags = model(idx_seq)
            print(true_tags)
            print(pred_tags)
            loss = loss_function(pred_tags, true_tags)
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
