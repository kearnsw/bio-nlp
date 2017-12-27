import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool1d
from torch.nn.functional import dropout as Dropout
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from bionlp.core.utils import load_emb, text2multichannel
from tqdm import *
import numpy as np
import sys
import os
import random
from argparse import ArgumentParser
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

torch.manual_seed(1)
random.seed(1986)


class CNN(nn.Module):
    def __init__(self, embeddings, filters, nb_classes, dropout, max_len, nb_filters):
        """
        Convolutional Neural Network (CNN) for Sentence Classification based on Kim 2014

        :param embeddings (list): a list of embedding matrices, embeddings are keyed by word indices
        :param filters (list): a list of ints corresponding to number of words to include per filter
        :param nb_classes (int): number of classes to make a prediction over
        :param dropout (float): percent of cells to knockout per epoch
        :param max_len (int): length of longest sequence
        """
        super().__init__()
        self.filters = filters
        self.nb_classes = nb_classes
        self.dropout = dropout
        self.nb_filters = nb_filters
        self.max_len = max_len
        self.embeddings = embeddings
        self.emb_dims = embeddings[0].shape[1]
        self.nb_channels = len(embeddings)

        # Ensure all embeddings are of the same dimensions
        assert all(embedding.shape[1] == self.emb_dims for embedding in self.embeddings)

        ################################################################################################################
        # Model Parameters
        ################################################################################################################

        # Create an embedding for each channel
        self.emb_layers = [nn.Embedding(embedding.shape[0], self.emb_dims, padding_idx=embedding.shape[0]-1)
                           for embedding in self.embeddings]

        # Create N convolutional layers equal to the number of filter variants, default = 3 (Kim 2014)
        # Each convolutional layer has form (in_channels, out_channels, kernel size) since we are dealing with a
        # 1-dimensional convolution and have flattened our input, we want our filter to be of size embedding x filter
        # size and to take a stride equal to the length of one word, i.e. the embedding dimension.
        self.conv1 = nn.ModuleList([nn.Conv1d(self.nb_channels, self.nb_filters, kernel_size * self.emb_dims,
                                              stride=self.emb_dims) for idx, kernel_size in enumerate(self.filters)])

        # Fully connected layer with output equal to the number of classes
        self.fc = nn.Linear(len(self.filters) * self.nb_filters, self.nb_classes)

        ################################################################################################################
        # Initialize Weights
        ################################################################################################################

        # Set weights to the embedding matrices, i.e. a matrix where each word embedding row corresponds to a word index
        for idx, emb in enumerate(self.emb_layers):
            emb.weight = nn.Parameter(torch.from_numpy(self.embeddings[idx]).float(), requires_grad=False)

        # Initialize the weights of the convolution layers
        for l in self.conv1:
            norm_const = l.kernel_size[0] * l.out_channels
            l.weight.data.normal_(0, np.sqrt(2./norm_const))

    def forward(self, sequence):
        """
        Forward propagation step

        :param sequence: an array of dimension (max_len, nb_channels) that contains the index of all words in sequence
        :return: prediction for each class
        """
        emb = [emb(sequence[channel]) for channel, emb in enumerate(self.emb_layers)]
        emb = torch.stack(emb).view(-1, self.nb_channels, self.emb_dims * self.max_len)  # flatten to 1d by nb_channels
        conv1 = [relu(conv(emb)) for conv in self.conv1]
        pool1 = [max_pool1d(conv, self.max_len - self.filters[idx] + 1).view(-1, self.nb_filters)
                 for idx, conv in enumerate(conv1)]
        combined = torch.cat(pool1, dim=1)
        dropout = Dropout(combined, self.dropout)
        return self.fc(dropout).view(self.nb_classes, -1)


def validate(model, valid_loader):
    model.eval()
    y_pred = []
    y_true = []
    for idx, mini_batch in enumerate(valid_loader):
        for question, true_type in list(zip(mini_batch[0], mini_batch[1])):
            class_predictions = model(question).data.numpy()
            prediction = np.argmax(class_predictions)
            y_true.append(true_type)
            y_pred.append(prediction)
    return accuracy_score(y_true, y_pred)


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
    cli_parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate for optimizer")
    cli_parser.add_argument("--train", action="store_true")
    cli_parser.add_argument("--validate", action="store_true")
    args = cli_parser.parse_args()

    ####################################################################################################################
    # Load Data
    ####################################################################################################################
    state_file = os.sep.join([args.models, "CNN.states"])
    model_file = os.sep.join([args.models, "CNN.model"])

    # Load embeddings
    emb_w2i = [load_emb(path, args.emb_dim) for path in args.word_emb.split(',')]
    embeddings = [emb for emb, w2i in emb_w2i]
    w2is = [w2i for emb, w2i in emb_w2i]

    # Load training data
    from bionlp.utils.Datasets import GARD
    dataset = GARD(args.text)
    max_len = max([len(q) for q in dataset.data])
    types = set(dataset.labels)
    type2idx = {type: idx for idx, type in enumerate(sorted(types))}

    # Convert strings to indices within data to speed up embedding layer
    data = list(zip(dataset.data, dataset.labels))
    random.shuffle(data)
    dataset.data, dataset.labels = zip(*data)
    dataset.data = [text2multichannel(sentence.split(), w2is, max_len=max_len) for sentence in dataset.data]
    dataset.labels = [type2idx[label] for label in dataset.labels]

    # Split training and dev data
    nb_examples = len(dataset)
    split_idx = round(.8 * nb_examples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx))))
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx,
                                                                                                          nb_examples))))

    ####################################################################################################################
    # Load Model
    ####################################################################################################################

    # Create model and set loss function and optimizer
    opts = {
        "embeddings": embeddings,
        "filters": [3, 4, 5],
        "nb_classes": 13,
        "dropout": 0.5,
        "max_len": max_len,
        "nb_filters": 100
    }

    model = CNN(**opts)
    type_weights = class_weight.compute_class_weight('balanced', np.unique(dataset.labels), dataset.labels)
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(type_weights))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True, mode="max")

    # Load checkpoint file if it exists
    if os.path.isfile(model_file):
        print("Loading model from {0}...".format(model_file))
        model = torch.load(model_file)

    elif os.path.isfile(state_file):
        print("Initializing models state from {0}".format(state_file))
        model.load_state_dict(torch.load(state_file))

    ####################################################################################################################
    # Train Model
    ####################################################################################################################
    # Store loss from each epoch for early stopping and plotting loss-curve
    if args.train:
        loss = np.zeros(args.epochs)

        # Train models using mini-batches
        print("Number of training examples:{0}".format(split_idx))
        for epoch in range(args.epochs):
            sys.stdout.write("Epoch {0}...\n".format(epoch))
            sys.stdout.flush()
            model.train()
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
            sys.stdout.flush()

            # Early Stopping
            if epoch > 0 and (loss[epoch - 1] - loss[epoch])/split_idx <= 0.00001:
                break

            val_loss = validate(model, valid_loader)
            print("Accuracy: {0}".format(val_loss))
            scheduler.step(val_loss)

            # Checkpoint
            torch.save(model.state_dict(), state_file, pickle_protocol=4)

        # Save best models
        torch.save(model, model_file, pickle_protocol=4)

    ####################################################################################################################
    # Validation
    ####################################################################################################################
    if args.validate:
        from sklearn.metrics import classification_report

        y_true = []
        y_pred = []
        idx2type = {v: k for k, v in type2idx.items()}

        for idx, mini_batch in enumerate(valid_loader):
            for question, true_type in list(zip(mini_batch[0], mini_batch[1])):
                class_predictions = model(question).data.numpy()
                prediction = np.argmax(class_predictions)
                y_true.append(true_type)
                y_pred.append(prediction)

        print(classification_report(y_true, y_pred))

