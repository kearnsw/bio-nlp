import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool1d
from torch.nn.functional import dropout as Dropout
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bionlp.core.utils import load_emb, text2seq
from bionlp.train.multiclass import train
import numpy as np
import os
import random
from argparse import ArgumentParser
from sklearn.utils import class_weight


torch.manual_seed(1)
random.seed(1986)


class CNN(nn.Module):
    def __init__(self, embedding, filters, nb_classes, dropout, max_len, nb_filters, multichannel=True):
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
        self.embedding = embedding
        self.vocab_size = embedding.shape[0]
        self.emb_dims = embedding.shape[1]
        self.nb_channels = 2 if multichannel else 1

        ################################################################################################################
        # Model Parameters
        ################################################################################################################

        # Create an embedding for each channel
        self.emb_layers = nn.ModuleList()
        self.static_emb = nn.Embedding(self.vocab_size, self.emb_dims, padding_idx=self.vocab_size-1)
        self.static_emb.weight = nn.Parameter(torch.from_numpy(self.embedding).float(), requires_grad=False)
        self.emb_layers.append(self.static_emb)
        if multichannel:
            self.nonstatic_emb = nn.Embedding(self.vocab_size, self.emb_dims, padding_idx=self.vocab_size-1)
            self.nonstatic_emb.weight = nn.Parameter(torch.from_numpy(self.embedding).float(), requires_grad=True)
            self.emb_layers.append(self.nonstatic_emb)

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

        # Initialize the weights of the convolution layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                norm_const = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2./norm_const))

    def forward(self, sequence):
        """
        Forward propagation step

        :param sequence: an array of dimension (max_len, nb_channels) that contains the index of all words in sequence
        :return: prediction for each class
        """
        emb = [l(sequence) for l in self.emb_layers]
        emb = torch.stack(emb).view(-1, self.nb_channels, self.emb_dims * self.max_len)  # flatten to 1d by nb_channels
        conv1 = [relu(conv(emb)) for conv in self.conv1]
        pool1 = [max_pool1d(conv, self.max_len - self.filters[idx] + 1).view(-1, self.nb_filters)
                 for idx, conv in enumerate(conv1)]
        combined = torch.cat(pool1, dim=1)
        dropout = Dropout(combined, self.dropout)
        return self.fc(dropout).view(self.nb_classes, -1)


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
    embedding, w2i = load_emb(args.word_emb, args.emb_dim)

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
    dataset.data = [text2seq(sentence.split(), w2i, max_len=max_len, autograd=False) for sentence in dataset.data]
    dataset.labels = [type2idx[label] for label in dataset.labels]

    # Split training and dev data
    nb_examples = len(dataset)
    split_idx = round(.95 * nb_examples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx))))
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx,
                                                                                                          nb_examples))))

    ####################################################################################################################
    # Load Model
    ####################################################################################################################

    # Create model and set loss function and optimizer
    opts = {
        "embedding": embedding,
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
    if args.train:
        train(model, optimizer, train_loader, valid_loader, args.nb_classes, loss_func, args.epochs,
              scheduler, state_file, model_file)
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
                class_predictions = model(Variable(question)).data.numpy()
                prediction = np.argmax(class_predictions)
                y_true.append(true_type)
                y_pred.append(prediction)

        print(classification_report(y_true, y_pred))

