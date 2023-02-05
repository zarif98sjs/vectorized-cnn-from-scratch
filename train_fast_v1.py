"""
CNN from scratch using only numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
import json
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import time
import pickle

## set seed for reproducibility
np.random.seed(120)

class Conv2D:
    def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weights = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * np.sqrt(1. / (self.kernel_size))
        # weights = np.random.randn(num_filters, in_channels, kernel_size, kernel_size)
        bias = np.zeros(num_filters) * np.sqrt(1. / (self.kernel_size))
        # bias = np.zeros(num_filters)

        self.trainable = True
        self.W = {"val": weights, "grad": np.zeros_like(weights)}
        self.b = {"val": bias, "grad": np.zeros_like(bias)}
        

        self.cache = None

    def __str__(self):
        return "Conv2D({}, {}, {}, {}, {})".format(self.in_channels, self.num_filters, self.kernel_size, self.stride, self.padding)

    def get_matrix_indices(self, x_shape):
        _ , C, H, W = x_shape
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        """
        creating the indexes needed to convert the matrix into consecutive columns
        """

        ## index i
        i0 = np.repeat(np.arange(self.kernel_size), self.kernel_size)
        i0 = np.tile(i0, C)
        all_levels = self.stride * np.repeat(np.arange(H_out), W_out)
        i = i0.reshape(-1, 1) + all_levels.reshape(1, -1)

        ## index j
        j0 = np.tile(np.arange(self.kernel_size), self.kernel_size * C)
        all_slides = self.stride * np.tile(np.arange(W_out), H_out)
        j = j0.reshape(-1, 1) + all_slides.reshape(1, -1)

        ## index d
        d = np.repeat(np.arange(C), self.kernel_size * self.kernel_size).reshape(-1, 1)

        return i, j, d

    def im2col(self, x):
        img = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        i, j, d = self.get_matrix_indices(x.shape)
        cols = img[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def col2im(self, dx_col, x_shape):
        N , C, H, W = x_shape
        H_pad = H + 2*self.padding
        W_pad = W + 2*self.padding

        X_pad = np.zeros((N, C, H_pad, W_pad))

        i, j, d = self.get_matrix_indices(x_shape)

        cols = np.array(np.hsplit(dx_col, N))

        np.add.at(X_pad, (slice(None), d, i, j), cols)

        if self.padding == 0:
            return X_pad
        return X_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
    def forward(self, x):

        N, _ , H_in, W_in = x.shape
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1

        X_col = self.im2col(x)
        W_col = self.W["val"].reshape(self.num_filters, -1)
        # print("W_col : ", W_col)
        b_col = self.b["val"].reshape(-1, 1)

        ## dot product
        out = W_col @ X_col + b_col

        out = np.array(np.hsplit(out, N)).reshape(N, self.num_filters, H_out, W_out)

        self.cache = (x, X_col, W_col)

        return out

    def backward(self, d_out):
        x, X_col, W_col = self.cache
        N = x.shape[0]

        # bias gradient: sum over all the dimensions except the channel dimension
        db = np.sum(d_out, axis=(0, 2, 3))

        # reshape dout
        # dout: (N, C, H, W) -> (N * C, H * W)
        d_out = d_out.reshape(d_out.shape[0] * d_out.shape[1], d_out.shape[2] * d_out.shape[3])
        d_out = np.array(np.vsplit(d_out, N))
        d_out = np.concatenate(d_out, axis=1)

        dW_col = d_out @ X_col.T        
        dX_col = W_col.T @ d_out

        # print("dX_col shape: ", dX_col.shape)
        dX = self.col2im(dX_col, x.shape)

        dW = dW_col.reshape(self.num_filters, self.in_channels, self.kernel_size, self.kernel_size)

        self.W["grad"] = dW
        self.b["grad"] = db

        return dX

class MaxPool2D:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None
        self.method = None

        self.trainable = False

    def __str__(self):
        return "MaxPool2D({}, {})".format(self.kernel_size, self.stride)

    def get_matrix_indices(self, x_shape):
        _ , C, H, W = x_shape
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        """
        creating the indexes needed to convert the matrix into consecutive columns
        """

        ## index i
        i0 = np.repeat(np.arange(self.kernel_size), self.kernel_size)
        i0 = np.tile(i0, C)
        all_levels = self.stride * np.repeat(np.arange(H_out), W_out)
        i = i0.reshape(-1, 1) + all_levels.reshape(1, -1)

        ## index j
        j0 = np.tile(np.arange(self.kernel_size), self.kernel_size * C)
        all_slides = self.stride * np.tile(np.arange(W_out), H_out)
        j = j0.reshape(-1, 1) + all_slides.reshape(1, -1)

        ## index d
        d = np.repeat(np.arange(C), self.kernel_size * self.kernel_size).reshape(-1, 1)

        return i, j, d

    def im2col(self, x):
        img = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        i, j, d = self.get_matrix_indices(x.shape)
        cols = img[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def col2im(self, dx_col, x_shape):
        N , C, H, W = x_shape
        H_pad = H + 2*self.padding
        W_pad = W + 2*self.padding

        X_pad = np.zeros((N, C, H_pad, W_pad))

        i, j, d = self.get_matrix_indices(x_shape)

        cols = np.array(np.hsplit(dx_col, N))

        np.add.at(X_pad, (slice(None), d, i, j), cols)

        if self.padding == 0:
            return X_pad
        return X_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def forward(self, x):
        _ , _ , H, W = x.shape
        fast_possible = (self.kernel_size == self.stride) or (H % self.kernel_size == 0 and W % self.kernel_size == 0)
        if fast_possible:
            self.method = 'faster'
            return self.forward_faster(x)
        else:
            self.method = 'fast'
            return self.forward_fast(x)


    def forward_fast(self, x):
        N, C , H_in, W_in = x.shape
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1

        X_split = x.reshape(N * C, 1, H_in, W_in)
        X_col = self.im2col(X_split)

        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, np.arange(max_idx.size)]
        out = out.reshape(N, C, H_out, W_out)
        self.cache = (x, X_col, max_idx)
        return out

    def forward_faster(self, x):
        N, C, H_in, W_in = x.shape
        
        assert self.kernel_size == self.stride, "kernel size must be equal to stride for fast implementation"
        assert H_in % self.kernel_size == 0, "height must be divisible by kernel size"
        assert W_in % self.kernel_size == 0, "width must be divisible by kernel size"

        H_out = H_in // self.kernel_size
        W_out = W_in // self.kernel_size

        x_split = x.reshape(N, C, H_out, self.kernel_size, W_out, self.kernel_size)
        out = x_split.max(axis=(3, 5))
        self.cache = (x, x_split , out)
        return out

    def backward(self, d_out):
        if self.method == 'fast':
            return self.backward_fast(d_out)
        elif self.method == 'faster':
            return self.backward_faster(d_out)
        return None

    def backward_fast(self, d_out):
        x, X_col, max_idx = self.cache
        N , C, H_in, W_in = x.shape

        # print("X_col shape: ", X_col.shape)

        d_out = d_out.reshape(1, -1)
        # print("d_out shape: ", d_out.shape)
        dX_col = np.zeros_like(X_col)
        
        dX_col[max_idx, np.arange(max_idx.size)] = d_out
        # print("dX_col shape: ", dX_col.shape)
        dX = self.col2im(dX_col, (N * C, 1, H_in, W_in))
        dX = dX.reshape(x.shape)

        return dX 

    def backward_faster(self, d_out):
        x, x_split, out = self.cache
        dx_split = np.zeros_like(x_split)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_split == out_newaxis)
        dout_newaxis = d_out[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_split)
        dx_split[mask] = dout_broadcast[mask]
        dx_split /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_split.reshape(x.shape)
        return dx

class Flatten:
    def __init__(self):
        self.cache = None
        self.trainable = False

    def __str__(self):
        return "Flatten"

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.cache)

class Dense:
    def __init__(self, out_features, in_features=None):
        self.out_features = out_features
        self.in_features = in_features
        W = None
        b = None

        self.trainable = True
        self.W = {"val": W, "grad": 0}
        self.b = {"val": b, "grad": 0}

        self.cache = None

    def __str__(self):
        return "Dense({})".format(self.out_features)
        
    def forward(self, x):
        if self.W["val"] is None:
            self.in_features = x.shape[1]
            self.W["val"] = np.random.randn(self.out_features, self.in_features) * np.sqrt(1.0 / self.in_features)
            self.b["val"] = np.random.randn(1, self.out_features) * np.sqrt(1.0 / self.in_features)
        out = x.dot(self.W["val"].T) + self.b["val"]
        self.cache = x
        return out

    def backward(self, d_out):
        x = self.cache
        N = x.shape[0]
        
        dW = d_out.T.dot(x) / N
        db = np.sum(d_out, axis=0, keepdims=True) / N

        dx = d_out.dot(self.W["val"])

        self.W["grad"] = dW
        self.b["grad"] = db

        return dx

class ReLU:
    def __init__(self):
        self.cache = None
        self.trainable = False

    def __str__(self):
        return "ReLU"

    def forward(self, x):
        out = np.maximum(0, x)
        self.cache = x
        return out

    def backward(self, d_out):
        x = self.cache
        d_out[x <= 0] = 0
        return d_out

class Softmax:
    def __init__(self):
        self.trainable = False
        pass
    
    def __str__(self):
        return "Softmax"

    def forward(self, x):
        # print("<< Softmax forward >>")
        # print(" x ", x)
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        # v = np.exp(x - np.max(x))
        # return v / np.sum(v, axis=1, keepdims=True)
        # exps = np.exp(x - np.max(x))
        # print(np.sum(exps, axis=1))
        # return exps / np.sum(exps, axis=1)[:, np.newaxis]

    def backward(self, d_out):
        return d_out



class AdamGD():

    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params
        
        self.momentum = {}
        self.rmsprop = {}

        # print(self.params)

        for key in self.params:
            print(key)
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):
        
        for key in self.params:
            # Momentum update.
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['d' + key] 
            # RMSprop update.
            self.rmsprop['sd' + key] =  (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (grads['d' + key]**2)
            # Update parameters.
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)  

        return self.params


class CNNModel():
    def __init__(self):
        self.layers = []
        self.load_layer_from_json("model.json")

    def load_layer_from_json(self, file_name):
        with open (file_name, "r") as f:
            layer_params = json.load(f)

        for layer in layer_params["model"]:
            if "Conv2D" in layer.keys():
                self.layers.append(Conv2D(layer["Conv2D"]["in_channels"], layer["Conv2D"]["num_filters"], layer["Conv2D"]["kernel_size"], layer["Conv2D"]["stride"], layer["Conv2D"]["padding"]))
            elif "ReLU" in layer.keys():
                self.layers.append(ReLU())
            elif "MaxPool2D" in layer.keys():
                self.layers.append(MaxPool2D(layer["MaxPool2D"]["kernel_size"], layer["MaxPool2D"]["stride"]))
            elif "Flatten" in layer.keys():
                self.layers.append(Flatten())
            elif "Dense" in layer.keys():
                self.layers.append(Dense(layer["Dense"]["out_features"]))
            elif "Softmax" in layer.keys():
                self.layers.append(Softmax())
    
        for layer in self.layers:
            print(layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, delL):
        for layer in reversed(self.layers):
            delL = layer.backward(delL)
        
        gradients = {}
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                gradients['dW' + str(i+1)] = layer.W['grad']
                gradients['db' + str(i+1)] = layer.b['grad']
        return gradients

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                params['W' + str(i+1)] = layer.W['val']
                params['b' + str(i+1)] = layer.b['val']
        return params

    def set_params(self, params, lr):
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer.W['val'] -= lr * params['dW'+ str(i+1)]
                layer.b['val'] -= lr * params['db'+ str(i+1)]

    def save_model_weights_pickle(self, file_name):
        params = self.get_params()
        # print(params)
        with open(file_name, "wb") as f:
            pickle.dump(params, f)


    def load_model_weights_pickle(self, file_name):

        with open(file_name, "rb") as f:
            params_new = pickle.load(f)

        # for i, layer in enumerate(self.layers):
        #     if layer.trainable:
        #         print("layer w", layer.W['val'])
        #         print("layer b", layer.b['val'])
        #         break

        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer.W['val'] = params_new['W' + str(i+1)]
                layer.b['val'] = params_new['b' + str(i+1)]

        # for i, layer in enumerate(self.layers):
        #     if layer.trainable:
        #         print(">layer w", layer.W['val'])
        #         print(">layer b", layer.b['val'])
        #         break


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

class History:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []

    def add(self, loss,val_loss, val_acc, val_f1):
        self.train_loss.append(loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.val_f1.append(val_f1)

    def plot(self):
        plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.plot(self.val_acc, label='val_acc')
        plt.legend()
        plt.show()

def train(x, y, val_x, val_y):
    # x = np.random.randn(100, 1, 28, 28)
    # ## normalize
    # # x = (x - np.mean(x)) / np.std(x)
    # y = np.random.randint(0, 10, (100, 1))

    # val_x = np.random.randn(100, 1, 28, 28)
    # val_y = np.random.randint(0, 10, (100, 1))

    print("----------------")
    print(x.shape)
    print(y.shape)
    print(val_x.shape)
    print(val_y.shape)

    model = CNNModel()
    history = History()
    print("HERE")
    # optimizer = AdamGD(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, params=model.get_params())
    print("AND HERE")
    BATCH_SIZE = 32
    EPOCHS = 30
    lr = 0.001

    BREAK = False

    for epoch in range(EPOCHS):
        for i in tqdm( range(0, x.shape[0], BATCH_SIZE) ):
            x_batch = x[i:i+BATCH_SIZE]
            y_batch = y[i:i+BATCH_SIZE]

            y_pred = model.forward(x_batch)
            y_batch = np.eye(10)[y_batch.reshape(-1)] ## one-hot encoding

            # print("y_pred", y_pred)
            # print(y_pred.shape)
            ## print sum
            # print("SUM : ",np.sum(y_pred, axis=1))
            
            # print("==")
            
            # print("y_batch", y_batch)
            # print(y_batch.shape)
            # print("------------------")

            err = y_pred - y_batch

            gradients = model.backward(err)
            # model.set_params(optimizer.update_params(gradients))
            model.set_params(gradients, lr)

        ## save model weights with epoch number
        model.save_model_weights_pickle("model_weights_epoch_" + str(epoch) + ".pkl")
        
        ## load model weights with epoch number
        model.load_model_weights_pickle("model_weights_epoch_" + str(epoch) + ".pkl")

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


        #     BREAK = True
        #     if BREAK:
        #         break
        # if BREAK:
        #     break

class DataLoader:
    def __init__(self, label_paths):
        self.label_paths = label_paths

    def load(self):
        df = pd.DataFrame()
        for label_path in self.label_paths:
            df = df.append(pd.read_csv(label_path))
        print(df.head())
        print(df.tail())
        """
            filename           original filename  scanid  digit database name original contributing team database name
        0  a00000.png   Scan_58_digit_5_num_8.png      58      5                  BHDDB      Buet_Broncos    training-a
        1  a00001.png   Scan_73_digit_3_num_5.png      73      3                  BHDDB      Buet_Broncos    training-a
        2  a00002.png   Scan_18_digit_1_num_3.png      18      1                  BHDDB      Buet_Broncos    training-a
        3  a00003.png  Scan_166_digit_7_num_3.png     166      7                  BHDDB      Buet_Broncos    training-a
        4  a00004.png  Scan_108_digit_0_num_1.png     108      0                  BHDDB      Buet_Broncos    training-a
        """

        COUNT = 1000

        labels = df['digit'].values
        print(labels[:10])

        filenames = df['database name'].values + '/' + df['filename'].values
        print(filenames[:10])

        """
        SAMPLE COUNT = 1000
        """
        labels = labels[:COUNT]
        filenames = filenames[:COUNT]

        images = []
        for filename in filenames:
            image = cv2.imread('dataset/NumtaDB_with_aug/' + filename, cv2.IMREAD_COLOR)
            images.append(image)

        ## resize
        images = [cv2.resize(image, (28, 28)) for image in images]
        
        images = np.array(images)
        print(images.shape)

        ## adjust dimensions : (N, H, W, C) -> (N, C, H, W)
        images = np.transpose(images, (0, 3, 1, 2))

        ## normalize
        images = images / 255.0

        ## convert (N, C, H, W) -> (N, 1, H, W)
        # images = images[:, 0:1, :, :]
        # print(images.shape)

        ## change background color
        images = 1 - images


        """
        stratified split
        """

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        """
        print sample distribution
        """

        print("Train distribution")
        print(np.unique(y_train, return_counts=True))
        print("Test distribution")
        print(np.unique(y_test, return_counts=True))

        return X_train, X_test, y_train, y_test


def test(X_test, y_test):
    model = CNNModel()
    model.load_model_weights_pickle("model_weights_kaggle.pkl")

    y_pred = model.forward(X_test)
    y_true = np.eye(10)[y_test.reshape(-1)]
    print("Test Accuracy: ", accuracy(y_pred, y_true))
    print("Test F1 Score: ", macro_f1(y_pred, y_true))

if __name__ == '__main__':

    """
    Test LeNet with dummy data
    """

    # train()

    # model = CNNModel()

    # calculate time to load data in seconds
    start = time.time()
    dataloader = DataLoader(label_paths=['dataset/NumtaDB_with_aug/training-a.csv', 'dataset/NumtaDB_with_aug/training-b.csv', 'dataset/NumtaDB_with_aug/training-c.csv'])
    x, val_x, y, val_y = dataloader.load()
    end = time.time()
    print("Time taken to load data: ", end - start)

    # train(x, y, val_x, val_y)

    test(val_x, val_y)


    