import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from bionlp.core.utils import load_embeddings, generate_emb_matrix, text2multichannel
from tqdm import *
import numpy as np
import sys
import os
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
        self.embeddings = kwargs["embeddings"]
        self.emb_dims = kwargs["embeddings"][0].shape[1]
        self.filters = kwargs["filters"]
        self.nb_classes = kwargs["nb_classes"]
        self.nb_channels = len(kwargs["embeddings"])
        self.dropout = kwargs["dropout"]
        self.nb_filters = 100
        self.max_len = kwargs["max_len"]

        # Model Parameters

        # Ensure all embeddings are of the same dimensions
        assert all(embedding.shape[1] == self.emb_dims for embedding in self.embeddings)

        # Create an embedding for each channel
        self.emb_layers = [nn.Embedding(embedding.shape[0], self.emb_dims, padding_idx=embedding.shape[0]-1)
                           for embedding in self.embeddings]

        # Set weights to the embedding matrix, i.e. a dictionary of word_index keys mapped to word_embedding values
        for idx, emb in enumerate(self.emb_layers):
            emb.weight = nn.Parameter(torch.from_numpy(self.embeddings[idx]).float(), requires_grad=False)

        # Create N convolutional layers equal to the number of filter variants, default = 3 (Kim 2014)
        # Each convolutional layer has form (in_channels, out_channels, kernel size) since we are dealing with a
        # 1-dimensional convolution and have flattened our input, we want our filter to be of size embedding x filter
        # size and to take a stride equal to the length of one word, i.e. the embedding dimension.
        self.conv1 = nn.ModuleList([nn.Conv1d(nb_channels, self.nb_filters, kernel_size * self.emb_dims, stride=self.emb_dims)
                                    for idx, kernel_size in enumerate(self.filters)])

        #
        self.fc = nn.Linear(len(self.filters) * self.nb_filters, self.nb_classes)

        for l in self.conv1:
            n = l.kernel_size[0] * l.out_channels
            l.weight.data.normal_(0, np.sqrt(2./n))

    def forward(self, sentence):
        emb = [emb(sentence[channel])for channel, emb in enumerate(self.emb_layers)]
        combined = torch.stack(emb).view(-1, nb_channels, self.emb_dims * self.max_len)  # flatten to 1d
        convs = [func.relu(conv(combined)) for conv in self.conv1]
        pooled = [func.max_pool1d(conv, self.max_len - self.filters[idx] + 1).view(-1, self.nb_filters) for idx, conv in enumerate(convs)]
        combined = torch.cat(pooled, dim=1)
        dropout = func.dropout(combined, self.dropout)
        return self.fc(dropout).view(self.nb_classes, -1)


def load_data(GARD_FILE):
    with open(GARD_FILE, "r", encoding="utf-8") as f:
        xml_data = f.read()
        soup = BeautifulSoup(xml_data, "xml")

    return soup.find_all("SubQuestion")


def load_emb(path, emb_dims):
    if "webQA" in path:
        model = Word2Vec.load(path)
        embeddings, word2idx = generate_emb_matrix(model.wv, emb_dims)
    else:
        embeddings, word2idx = load_embeddings(path, emb_dims)
    return embeddings, word2idx


if __name__ == "__main__":

    cli_parser = ArgumentParser()
    cli_parser.add_argument("-e", "--word_emb", type=str, help="Comma separated list of paths to embedding files")
    cli_parser.add_argument("-d", "--emb_dim", type=int, help="Embedding dimensions")
    cli_parser.add_argument("--text", type=str, help="Directory with raw text files")
    cli_parser.add_argument("--epochs", type=int, help="Number of epochs")
    cli_parser.add_argument("--batch_size", type=int, default=10)
    cli_parser.add_argument("--nb_classes", type=int, default=13)
    cli_parser.add_argument("--cuda", action="store_true")
    cli_parser.add_argument("--models", type=str, help="model directory")
    cli_parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="learning rate for optimizer")
    args = cli_parser.parse_args()

    state_file = os.sep.join([args.models, "CNN.states"])
    model_file = os.sep.join([args.models, "CNN.model"])

    # Load embeddings
    emb_paths = [path for path in args.word_emb.split(',')]
    emb_w2i = [load_emb(path, args.emb_dim) for path in emb_paths]
    embeddings = [emb for emb, w2i in emb_w2i]
    w2is = [w2i for emb, w2i in emb_w2i]
    nb_channels = len(w2is)

    # Load training data
    from bionlp.utils.Datasets import GARD
    dataset = GARD(args.text)
    max_len = max([len(q) for q in dataset.data])
    types = set(dataset.labels)
    type2idx = {type: idx for idx, type in enumerate(sorted(types))}
    type_weights = class_weight.compute_class_weight('balanced', np.unique(dataset.labels), dataset.labels)

    # Convert strings to indices within data to speed up embedding layer
    dataset.data = [text2multichannel(sentence.split(), w2is, max_len=max_len) for sentence in dataset.data]
    dataset.labels = [type2idx[label] for label in dataset.labels]

    # Split training and dev data
    nb_examples = len(dataset.data)
    split_idx = round(.9 * nb_examples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx))))
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx,
                                                                                                          nb_examples))))

    # Create model and set loss function and optimizer
    opts = {
        "embeddings": embeddings,
        "filters": [3, 4, 5],
        "nb_classes": 13,
        "dropout": 0.5,
        "max_len": max_len
    }

    model = CNN(**opts)
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(type_weights))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate)

    # Load checkpoint file if it exists
    if os.path.isfile(state_file):
        print("Initializing models state from file...")
        model.load_state_dict(torch.load(state_file))

    # Store loss from each epoch for early stopping and plotting loss-curve
    loss = np.zeros(args.epochs)

    print("Number of training examples:{0}".format(split_idx))
    # Train models using mini-batches
    for epoch in range(args.epochs):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()

        for mini_batch in tqdm(train_loader, total=len(train_loader)):
            batch_loss = Variable(torch.FloatTensor([0]))  # Zero out the loss from last batch
            model.zero_grad()                              # Zero out the gradient from last batch
            for doc, label in list(zip(mini_batch[0], mini_batch[1])):
                class_pred = model(doc)
                batch_loss += loss_func(class_pred.view(-1, args.nb_classes), Variable(torch.LongTensor([label])))

            # Backpropagate the loss for each mini-batch
            batch_loss.backward()
            optimizer.step()
            loss[epoch] += batch_loss.data[0]

        sys.stdout.write("Loss: {0}\n".format(loss[epoch]/split_idx))
        # sys.stdout.write("Accuracy: {0}\n".format(accuracy_count/len(x_train)))
        sys.stdout.flush()

        # Early Stopping ** train with higher learning rate then lower the learning rate **
        if epoch > 0 and loss[epoch - 1] - loss[epoch] <= 0.0001:
            break

        # Checkpoint
        torch.save(model.state_dict(), state_file, pickle_protocol=4)

    # Save best models
    torch.save(model, model_file, pickle_protocol=4)

    from sklearn.metrics import classification_report

    idx2type = {v: k for k, v in type2idx.items()}

    y_true = []
    y_pred = []

    for idx, mini_batch in enumerate(valid_loader):
        questions = mini_batch[0]
        true_types = mini_batch[1]
        for question, true_type in list(zip(questions,true_types)):
            preds = model(question).data.numpy()
            pred_type = np.argmax(preds)
            y_true.append(true_type)
            y_pred.append(pred_type)

    print(classification_report(y_true, y_pred))

