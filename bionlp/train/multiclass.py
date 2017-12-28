import numpy as np
from tqdm import *
import sys
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


def validate(model, valid_loader):
    """
    Returns an accuracy metric for validation
    :param model: The model to be validated
    :param valid_loader: A DataLoader class containing the indexed validation examples and labels
    :return:
    """
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


def train(model, optimizer, train_loader, valid_loader, nb_classes, loss_func=CrossEntropyLoss(), epochs=100,
          scheduler=None, state_file="states.pkl", model_file="model.pkl"):
    """
    Train a model using backpropagation for a multi-class classification task

    :param model: A model that outputs a prediction over the number of classes
    :param optimizer: An optimizer to perform gradient descent
    :param train_loader: A DataLoader class containing the indexed training examples and labels
    :param valid_loader: A DataLoader class containing the indexed validation examples and labels
    :param nb_classes: Number of classes over which to make a prediction
    :param loss_func: (optional) Loss function for computing the loss for each batch, default is CrossEntropyLoss()
    :param epochs: (optional) Number of epochs to run gradient descent, default is 100
    :param scheduler: (optional) A scheduler class for updating the learning rate based on a validation metric
    :param state_file: Output path for checkpointing model parameters
    :param model_file: Output path for saving the final model
    :return:
    """
    # Store loss from each epoch for early stopping and plotting loss-curve
    loss = np.zeros(epochs)

    # Train models using mini-batches
    nb_examples = len(train_loader.dataset)
    print("Number of training examples:{0}".format(nb_examples))
    for epoch in range(epochs):
        sys.stdout.write("Epoch {0}...\n".format(epoch))
        sys.stdout.flush()
        model.train()
        for mini_batch in tqdm(train_loader, total=len(train_loader)):
            batch_loss = Variable(torch.FloatTensor([0]))  # Zero out the loss from last batch
            model.zero_grad()                              # Zero out the gradient from last batch

            for doc, label in list(zip(mini_batch[0], mini_batch[1])):
                class_pred = model(doc)
                batch_loss += loss_func(class_pred.view(-1, nb_classes), Variable(torch.LongTensor([label])))

            # Backpropagate the loss for each mini-batch
            batch_loss.backward()
            optimizer.step()
            loss[epoch] += batch_loss.data[0]

        sys.stdout.write("Loss: {0}\n".format(loss[epoch]/nb_examples))
        sys.stdout.flush()

        # Early Stopping
        if epoch > 0 and (loss[epoch - 1] - loss[epoch])/nb_examples <= 0.00001:
            break

        val_loss = validate(model, valid_loader)
        print("Accuracy: {0}".format(val_loss))
        if scheduler:
            scheduler.step(val_loss)

        # Checkpoint
        torch.save(model.state_dict(), state_file, pickle_protocol=4)

    # Save best models
    torch.save(model, model_file, pickle_protocol=4)
