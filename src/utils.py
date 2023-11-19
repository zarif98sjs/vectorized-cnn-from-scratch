import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import CNNModel
from history import History

def loss(y_pred, y_true):
    """
    y_pred: (N, C) array of predicted class scores
    y_true: (N, C) array of one-hot encoded true class labels
    """
    EPS = 1e-8
    N = y_pred.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + EPS)) / N
    return loss

def accuracy(y_pred, y_true):
    """
    y_pred: (N, C) array of predicted class scores
    y_true: (N, C) array of one-hot encoded true class labels
    """
    N = y_pred.shape[0]
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    acc = np.sum(y_pred == y_true) / N
    return acc * 100

def macro_f1(y_pred, y_true):
    """
    y_pred: (N, C) array of predicted class scores
    y_true: (N, C) array of one-hot encoded true class labels
    """
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1 * 100

def train(x, y, val_x, val_y):
    print("TRAINING STARTED")
    print("----------------")
    print(x.shape)
    print(y.shape)
    print(val_x.shape)
    print(val_y.shape)

    model = CNNModel()

    config = {
        'MODEL_DESCRIPTION': str(model),
        'BATCH_SIZE': 32,
        'EPOCHS': 1,
        'lr': 0.001,
    }

    print("MODEL CREATED")
    print(model)
    history = History()

    for epoch in range(config['EPOCHS']):
        for i in tqdm( range(0, x.shape[0],config['BATCH_SIZE'])):
            x_batch = x[i:i+config['BATCH_SIZE']]
            y_batch = y[i:i+config['BATCH_SIZE']]

            y_pred = model.forward(x_batch)
            y_batch = np.eye(10)[y_batch.reshape(-1)] ## one-hot encoding
            err = y_pred - y_batch
            gradients = model.backward(err)
            model.set_params(gradients, config['lr'])

        ## save model weights with epoch number
        model.save_model_weights_pickle("model_weights_epoch_" + str(epoch) + ".pkl")

        """
        Loss
        """
        train_loss = loss(y_pred, y_batch)

        """
        Validation
        """
        val_y_pred = model.forward(val_x)
        val_y_true = np.eye(10)[val_y.reshape(-1)]
        val_loss = loss(val_y_pred, val_y_true)
        val_acc = accuracy(val_y_pred, val_y_true)
        val_f1 = macro_f1(val_y_pred, val_y_true)

        """
        Save history
        """
        history.add(train_loss, val_loss, val_acc, val_f1)

        """
        Log
        """
        print("Epoch: {}, Train Loss: {}, Val Loss: {}, Val Acc: {}, Val F1: {}".format(epoch, train_loss, val_loss, val_acc, val_f1))


    history.save("history.pkl")
    history.plot()

    ## save config as json
    with open("config.json", "w") as f:
        json.dump(config, f)


def test(X_test, y_test):
    model = CNNModel()
    model.load_model_weights_pickle("model_weights_kaggle.pkl")

    y_pred = model.forward(X_test)
    y_true = np.eye(10)[y_test.reshape(-1)]
    print("Test Accuracy: ", accuracy(y_pred, y_true))
    print("Test F1 Score: ", macro_f1(y_pred, y_true))

def predict(X_test, image_names):
    model = CNNModel()
    model.load_model_weights_pickle("1705010_model.pkl")

    y_pred = model.forward(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    """
    FileName,Digit
    """
    df = pd.DataFrame()
    df['FileName'] = image_names
    df['Digit'] = y_pred
    df.to_csv("predictions.csv", index=False)
