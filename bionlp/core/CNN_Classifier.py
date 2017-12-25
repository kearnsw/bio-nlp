import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from bionlp.core.utils import text2seq, idx_tags, tags2idx, load_embeddings, plot_loss, generate_emb_matrix
from bionlp.core.utils import create_batches
from tqdm import *
import numpy as np
import sys
import os
import re
import random
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.utils import class_weight

torch.manual_seed(1)
random.seed(1986)


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.word2idx = kwargs["word2idx"]
        self.vocab_size = len(self.word2idx) + 2  # Plus two to include <UNK> and <PAD>
        self.emb_dims = kwargs["embedding_matrix"].shape[1]
        self.filters = kwargs["filters"]
        self.nb_classes = kwargs["nb_classes"]
        self.embedding_matrix = kwargs["embedding_matrix"]
        self.dropout = kwargs["dropout"]
        self.nb_filters = 100
        self.max_len = kwargs["max_len"]

        # Model Layers
        # Embedding layer holds a dictionary of word_index keys mapped to word_embedding values
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dims, padding_idx=self.vocab_size - 1)
        self.embedding.weight = nn.Parameter(torch.from_numpy(self.embedding_matrix).float(),
                                             requires_grad=True)

        # Create N convolutional layers equal to the number of filter variants, default = 3 (Kim 2014)
        # Each convolutional layer is of dimension 1 x
        self.conv_layers = [nn.Conv1d(1, self.nb_filters, kernel_size * self.emb_dims, stride=self.emb_dims)
                            for idx, kernel_size in enumerate(self.filters)]

        #
        self.conv2class = nn.Linear(len(self.filters) * self.nb_filters, self.nb_classes)

        for l in self.conv_layers:
            n = l.kernel_size[0] * l.out_channels
            l.weight.data.normal_(0, np.sqrt(2./n))

    def forward(self, sentence):
        emb = self.embedding(sentence).view(-1, 1, self.emb_dims * self.max_len)  # flatten to 1d
        convs = [func.relu(conv(emb)) for conv in self.conv_layers]
        pooled = [func.max_pool1d(conv, self.max_len - self.filters[idx] + 1).view(-1, self.nb_filters) for idx, conv in enumerate(convs)]
        combined = torch.cat(pooled, dim=1)
        dropout = func.dropout(combined, self.dropout)
        return self.conv2class(dropout).view(self.nb_classes, -1)


def load_data(GARD_FILE):
    with open(GARD_FILE, "r", encoding="utf-8") as f:
        xml_data = f.read()
        soup = BeautifulSoup(xml_data, "xml")

    return soup.find_all("SubQuestion")


if __name__ == "__main__":

    cli_parser = ArgumentParser()
    cli_parser.add_argument("-e", "--word_emb", type=str, help="Path to the binary embedding file")
    cli_parser.add_argument("-d", "--emb_dim", type=int, help="Embedding dimensions")
    cli_parser.add_argument("--text", type=str, help="Directory with raw text files")
    cli_parser.add_argument("--epochs", type=int, help="Number of epochs")
    cli_parser.add_argument("--batch_size", type=int, default=10)
    cli_parser.add_argument("--nb_classes", type=int, default=13)
    cli_parser.add_argument("--cuda", action="store_true")
    cli_parser.add_argument("--models", type=str, help="model directory")
    cli_parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate for optimizer")
    args = cli_parser.parse_args()

    state_file = os.sep.join([args.models, "CNN.states"])
    model_file = os.sep.join([args.models, "CNN.model"])

    # Load training data and split into training and dev

    questions = load_data(args.text)
    types = set([q["qt"] for q in questions])
    type2idx = {type: idx for idx, type in enumerate(sorted(types))}
    data = [(q.text, q["qt"]) for q in questions]
    random.shuffle(data)
    posts = [q for q, qt in data]
    labels = [qt for q, qt in data]
    max_len = max([len(post) for post in posts])

    # Split training and dev data
    split_idx = round(.9 * len(posts))
    x_train = posts[:split_idx]
    y_train = labels[:split_idx]
    x_dev = posts[split_idx:]
    y_dev = labels[split_idx:]

    # Convert sentences to embeddings and question types to class indices
    if "webQA" in args.word_emb:
        model = Word2Vec.load(args.word_emb)
        embeddings, word2idx = generate_emb_matrix(model.wv, 100)
    else:
        embeddings, word2idx = load_embeddings(args.word_emb, args.emb_dim)

    # Convert data into sequence using indices from embeddings and padding the sequences
    x_train = [text2seq(sentence.split(), word2idx, max_len=max_len) for sentence in x_train]
    y_train = [type2idx[qt] for qt in y_train]
    x_dev = [text2seq(sentence.split(), word2idx, max_len=max_len) for sentence in x_dev]
    y_dev = [type2idx[qt] for qt in y_dev]

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    y_train = [Variable(torch.LongTensor([y])) for y in y_train]

    # Create model and set loss function and optimizer
    opts = {
        "word2idx": word2idx,
        "embedding_matrix": embeddings,
        "filters": [3, 4, 5],
        "nb_classes": 13,
        "dropout": 0.5,
        "max_len": max_len
    }

    model = CNN(**opts)
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    # Create files to store intermediate model states and the final best model
    try:
        emb_version = re.match(r".*(?=\.[^/.]+)", os.path.basename(args.word_emb))[0]
    except:
        print("failed to extract embedding name, expected filename of type filename.* (e.g. glove.6B.50d.txt or "
              "PubMed-win-2.bin)")
        emb_version = ""

    # Load checkpoint file if it exists
    if os.path.isfile(state_file):
        print("Initializing models state from file...")
        model.load_state_dict(torch.load(state_file))

    # Store loss from each epoch for early stopping and plotting loss-curve
    loss = np.zeros(args.epochs)

    # Create batches
    mini_batches = create_batches(list(zip(x_train, y_train)), args.batch_size)

    print("Number of training examples:{0}".format(len(y_train)))
    # Train models using mini-batches
    for epoch in range(args.epochs):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for mini_batch in tqdm(mini_batches, total=len(mini_batches)):
            batch_loss = Variable(torch.FloatTensor([0]))  # Zero out the loss from last batch
            model.zero_grad()                              # Zero out the gradient from last batch
            for doc, label in mini_batch:
                class_pred = model(doc)
                batch_loss += loss_func(class_pred.view(-1, args.nb_classes), label)

            # Backpropagate the loss for each mini-batch
            batch_loss.backward()
            optimizer.step()
            loss[epoch] += batch_loss.data[0]

        sys.stdout.write("Loss: {0}\n".format(loss[epoch]/len(x_train)))
        # sys.stdout.write("Accuracy: {0}\n".format(accuracy_count/len(x_train)))
        sys.stdout.flush()

        # Early Stopping ** train with higher learning rate then lower the learning rate **
        if epoch > 0 and loss[epoch - 1] - loss[epoch] <= 0.0001:
            break

        # Checkpoint
        torch.save(model.state_dict(), state_file, pickle_protocol=4)

    # Save best models
    torch.save(model, model_file, pickle_protocol=4)
