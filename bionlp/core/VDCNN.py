import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool1d
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bionlp.core.utils import load_emb, text2seq, kmax_pooling
from bionlp.train.multiclass import train
import numpy as np
import os
import random
from argparse import ArgumentParser
from sklearn.utils import class_weight
from math import floor
from torch.autograd import Variable


class VDCNN(nn.Module):

    def __init__(self, embedding, blocks):
        """
        Very Deep Convolutional Network based on Conneau 2017 with the modification of using pre-trained embeddings,
        since biomedical data sets tend to be too small to see improved performance from character level embeddings.

        :param embedding:
        """
        super().__init__()
        self.embedding = embedding
        self.vocab_size = embedding.shape[0]
        self.emb_dims = embedding.shape[1]
        self.blocks = blocks
        self.nb_filters = 64
        self.kernel_size = 3
        self.padding = floor(self.kernel_size/2)

        # Model layers
        self.emb = nn.Embedding(self.vocab_size, self.emb_dims, padding_idx=self.vocab_size - 1)
        self.conv0 = nn.Conv1d(self.emb_dims, self.nb_filters, self.kernel_size, padding=self.padding)
        self.conv1 = ConvBlock(self.nb_filters, self.nb_filters, 3)
        self.conv2 = ConvBlock(self.nb_filters, self.nb_filters, 3)
        self.conv3 = ConvBlock(self.nb_filters, self.nb_filters*2, 3)
        self.conv4 = ConvBlock(self.nb_filters*2, self.nb_filters*2, 3)
        self.conv5 = ConvBlock(self.nb_filters*2, self.nb_filters*4, 3)
        self.conv6 = ConvBlock(self.nb_filters*4, self.nb_filters*4, 3)
        self.conv7 = ConvBlock(self.nb_filters*4, self.nb_filters*8, 3)
        self.conv8 = ConvBlock(self.nb_filters*8, self.nb_filters*8, 3)

        self.fc1 = nn.Linear(8*128, 1024)
        self.fc2 = nn.Linear(1024, 13)

        # Initialize the weights of the convolution layers
        for m in self.modules():
            if type(m) == nn.modules.conv.Conv1d:
                norm_const = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / norm_const))
            elif type(m) == nn.modules.sparse.Embedding:
                m.weight = nn.Parameter(torch.from_numpy(self.embedding).float(), requires_grad=False)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.5)

    def forward(self, text):
        emb = self.emb(text)
        conv0 = self.conv0.forward(emb.view(-1, self.emb_dims, len(text)))

        conv1 = self.conv1.forward(conv0)
        conv2 = self.conv2.forward(conv1)
        pool1 = max_pool1d(conv2, 3, stride=2, padding=1)

        conv3 = self.conv3.forward(pool1)
        conv4 = self.conv4.forward(conv3)

        """
        pool2 = max_pool1d(conv4, 3, stride=2, padding=1)
        conv5 = self.conv5.forward(pool2)
        conv6 = self.conv6.forward(conv5)
        pool3 = max_pool1d(conv6, 3, stride=2, padding=1)
        conv7 = self.conv7.forward(pool3)
        conv8 = self.conv8.forward(conv7)
        """

        kpool = kmax_pooling(conv4, -1, 8)
        fc1 = self.fc1.forward(kpool.view(-1, 128*8))
        fc2 = self.fc2.forward(relu(fc1))
        return relu(fc2)


class ConvBlock(nn.Module):

    def __init__(self, input_dim, nb_filters, kernel_size):
        super().__init__()
        self.padding = floor(kernel_size/2)
        self.conv1 = nn.Conv1d(input_dim, nb_filters, kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(nb_filters, nb_filters, kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(nb_filters)
        self.bn2 = nn.BatchNorm1d(nb_filters)

    def forward(self, _input):
        conv1 = self.conv1(_input)
        a_1 = relu(self.bn1(conv1))
        conv2 = self.conv2(a_1)
        return relu(self.bn2(conv2))


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
    state_file = os.sep.join([args.models, "VDCNN.states"])
    model_file = os.sep.join([args.models, "VDCNN.model"])

    # Load embeddings
    emb, w2i = load_emb(args.word_emb, args.emb_dim)

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
    split_idx = round(.9 * nb_examples)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx))))
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(list(range(split_idx,
                                                                                                          nb_examples))))

    ####################################################################################################################
    # Load Model
    ####################################################################################################################

    # Create model and set loss function and optimizer
    opts = {
        "embedding": emb,
        "blocks": []
    }

    model = VDCNN(**opts)
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

