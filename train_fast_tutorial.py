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
        b_col = self.b["val"].reshape(-1, 1)

        print("X_col.shape: ", X_col.shape)
        print("W_col.shape: ", W_col.shape)
        print("b_col.shape: ", b_col.shape)

        ## dot product
        out = W_col @ X_col + b_col

        print("out.shape: ", out.shape)

        out = np.array(np.hsplit(out, N)).reshape(N, self.num_filters, H_out, W_out)

        print("out.shape: ", out.shape)

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
        print("x_shape :",x.shape)
        x_split = x.reshape(N, C, H_out, self.kernel_size, W_out, self.kernel_size)
        print("x_split shape: ",x_split.shape)
        print("x_split[0]: ",x_split[0])
        temp_x_split = x_split.max(axis=5)
        print("temp_x_split shape: ",temp_x_split.shape)
        print("temp_x_split[0]: ",temp_x_split[0])
        out = x_split.max(axis=(3, 5))
        self.cache = (x, x_split , out)
        return out

    def backward(self, d_out):
        if self.method == 'fast':
            return self.backward_fast(d_out)
        elif self.method == 'faster':
            return self.backward_faster(d_out)
        elif self.method == 'slow':
            return self.backward_slow(d_out)
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

    def backward_slow(self, dout):
        """
            Distributes error through max pooling layer.
            Parameters:
            - dout: Previous layer with the error.
            Returns:
            - dX: Conv layer updated with error.

            We need to distribute the error to the correct input element.
            First, we find the index responsible for the maximum value in the input.
            The gradient dout[i, c, h, w] is backpropagated only to that corresponding index in dX.
        """
        X, _, _ = self.cache
        N, C, H_in, W_in = dout.shape
        dX = np.zeros(X.shape)

        for i in range(N): # For each image.

            for c in range(C): # For each channel.

                for h in range(H_in): # Slide the filter vertically.
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size

                    for w in range(W_in): # Slide the filter horizontally.
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        region = X[i, c, h_start:h_end, w_start:w_end]
                        max_index = np.unravel_index(np.argmax(region), region.shape)
                        max_x, max_y = max_index

                        dX[i, c, h_start:h_end, w_start:w_end][max_x, max_y] = dout[i, c, h, w]

        return dX 

    def backward_faster(self, d_out):
        print("dout_shape [maxpool] : ",d_out.shape)
        print("dout [maxpool] : ",d_out[0])
        x, x_split, out = self.cache
        print("x_split shape [maxpool] : ",x_split.shape)
        print("x_split [maxpool] : ",x_split[0])
        print("out shape [maxpool] : ",out.shape)
        dx_split = np.zeros_like(x_split)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        print("out_newaxis shape [maxpool] : ",out_newaxis.shape)
        print("out_newaxis [maxpool] : ",out_newaxis[0])
        mask = (x_split == out_newaxis)
        print("mask shape : ",mask.shape)
        print("mask ",mask[0])
        dout_newaxis = d_out[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_split)
        print("dout_broadcast shape [maxpool] : ",dout_broadcast.shape)
        print("dout_broadcast [maxpool] : ",dout_broadcast[0])
        dx_split[mask] = dout_broadcast[mask]
        print("dx_split shape [maxpool] : ",dx_split.shape)
        print("dx_split [maxpool] : ",dx_split[0])
        dx_split /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_split.reshape(x.shape)
        return dx

    # def backward_faster(self, d_out):
    #     """
    #     Reference: 
    #     https://stackoverflow.com/questions/61954727/max-pooling-backpropagation-using-numpy
    #     https://gitlab.cs.washington.edu/liangw6/assignment2-for-stanford231n

    #     "This implementation has a crucial (but often ignored) mistake: 
    #     in case of multiple equal maxima, it backpropagates to all of them which can easily result in 
    #     vanishing / exploding gradients / weights. You can propagate to (any) one of the maximas, not all of them. 
    #     tensorflow chooses the first maxima."

    #     """
    #     """
    #     In this backward pass, the gradient with respect to the input x is computed, given the gradient with respect to the output d_out.
    #     For this efficient implementation, in the forward pass, the input x is divided into non-overlapping regions and the maximum value in each region is selected as the output.
    #     And in the backward pass, the gradient is propagated back to the input x in a way that only the elements of x corresponding to the maximum value in each region receive the gradient.
    #     """
    #     x, x_split, out = self.cache
    #     dx_split = np.zeros_like(x_split)
    #     """
    #     The purpose of mask is to identify which elements of x correspond to the maximum value in each region, and then distribute the gradient to these elements.
    #     """
    #     mask = (x_split == np.expand_dims(np.expand_dims(out, 3), 5))
    #     dout_broadcast, _ = np.broadcast_arrays(np.expand_dims(np.expand_dims(d_out, 3), 5), dx_split)
    #     """
    #     This mask is then used to update the values of dx_split. 
    #     The values of dx_split are updated by taking d_out of the mask (after boradcasting), and then normalizing by the sum of the mask along the 3rd and 5th dimensions.
    #     The 3rd and 5th dimensions are from this shape: (N, C, H_out, F, W_out, F)
    #     """
    #     dx_split[mask] = dout_broadcast[mask]
    #     dx_split /= np.sum(mask, axis=(3, 5), keepdims=True)
    #     dx = dx_split.reshape(x.shape)
    #     return dx

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

        print("d_out shape: ", d_out.shape)
        print("x shape: ", x.shape)
        
        dW = d_out.T.dot(x) / N
        print("dW shape: ", dW.shape)
        db = np.sum(d_out, axis=0, keepdims=True) / N
        print("db shape: ", db.shape)

        print("W shape: ",self.W["val"].shape)
        dx = d_out.dot(self.W["val"])
        print("dx shape: ", dx.shape)

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
        print("x shape: ", x.shape)
        print("x[0] : ", x[0])
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

class CNNModel():
    def __init__(self):
        self.layers = []
        self.load_layer_from_json("model.json")

    def __str__(self):
        ret = ""
        for layer in self.layers:
            ret += str(layer) + "\n"
        return ret

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
    
        # for layer in self.layers:
        #     print(layer)
        
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

## History dictionary to store the training and validation loss and accuracy
class History:
    def __init__(self):
        self.history = {'loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    def add(self, loss, val_loss, val_acc, val_f1):
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
    
    def plot(self):
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['loss'], label='train_loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig('loss.png')
        plt.clf()
        # plt.show()

        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig('acc.png')
        plt.clf()
        # plt.show()

        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['val_f1'], label='val_f1')
        plt.legend()
        plt.savefig('f1.png')
        # plt.show()
        plt.clf()

        """
        plot all in same fig
        """
        
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['loss'], label='train_loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.plot(np.array(self.history['val_acc'])/100, label='val_acc')
        plt.plot(np.array(self.history['val_f1'])/100, label='val_f1')
        plt.legend()
        plt.savefig("all_metric_in_one.png")
        # plt.show()
        plt.clf()

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.history, f)

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
            model.set_params(gradients, config['lr'])

        ## save model weights with epoch number
        model.save_model_weights_pickle("model_weights_epoch_" + str(epoch) + ".pkl")
        
        ## load model weights with epoch number
        # model.load_model_weights_pickle("model_weights_epoch_" + str(epoch) + ".pkl")

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


class DataLoader:
    def __init__(self, label_paths):
        self.label_paths = label_paths

    def load(self, isTestSet=False):
        df = pd.DataFrame()
        for label_path in self.label_paths:
            df = df.append(pd.read_csv(label_path))
        # print(df.head())
        # print(df.tail())
        print(df.shape)
        """
            filename           original filename  scanid  digit database name original contributing team database name
        0  a00000.png   Scan_58_digit_5_num_8.png      58      5                  BHDDB      Buet_Broncos    training-a
        1  a00001.png   Scan_73_digit_3_num_5.png      73      3                  BHDDB      Buet_Broncos    training-a
        2  a00002.png   Scan_18_digit_1_num_3.png      18      1                  BHDDB      Buet_Broncos    training-a
        3  a00003.png  Scan_166_digit_7_num_3.png     166      7                  BHDDB      Buet_Broncos    training-a
        4  a00004.png  Scan_108_digit_0_num_1.png     108      0                  BHDDB      Buet_Broncos    training-a
        """

        COUNT = 1000
        if isTestSet:
            COUNT = 50000

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

        if isTestSet:
            labels = labels.reshape(-1, 1)
            return images, labels


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

    NUM_CHANNELS = 1
    NUM_FILTERS = 1
    IMG_SIZE = 5
    NUM_CLASS = 10
    NUM_SAMPLES = 100

    ## dummy data
    X_train = np.random.randint(0, NUM_CLASS, (NUM_SAMPLES, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    y_train = np.random.randint(0, NUM_CLASS, (NUM_SAMPLES, NUM_CHANNELS))

    print(X_train[0])
    print(X_train[0].shape)
    print(y_train[0])

    """
    MAXPOOL F/W FASTER TEST
    """

    # maxpool = MaxPool2D(kernel_size=2, stride=2)
    # x = maxpool.forward(X_train)
    # print(x.shape)
    # print(x[0])

    """
    FULL F/W B/W TEST
    """

    conv2d = Conv2D(in_channels=NUM_CHANNELS, num_filters=NUM_FILTERS, kernel_size=2)
    relu = ReLU()
    maxpool = MaxPool2D(kernel_size=2, stride=2)
    flatten = Flatten()
    dense = Dense(out_features=10)
    softmax = Softmax()

    x = conv2d.forward(X_train)
    print(x.shape)
    print(x[0])

    x = relu.forward(x)
    print(x.shape)
    print(x[0])

    x = maxpool.forward(x)
    print(x.shape)
    print(x[0])

    x = flatten.forward(x)
    print(x.shape)
    print(x[0])

    x = dense.forward(x)
    print(x.shape)
    print(x[0])

    x = softmax.forward(x)
    print(x.shape)
    print(x[0])

    y_pred = x
    y_true = np.eye(10)[y_train.reshape(-1)]

    loss = loss(y_pred, y_true)
    print(loss)

    err = y_pred - y_true
    print(err.shape)
    print(err[0])

    err = softmax.backward(err)
    print(err.shape)
    print(err[0])

    err = dense.backward(err)
    print(err.shape)
    print(err[0])

    err = flatten.backward(err)
    print(err.shape)
    print(err[0])

    err = maxpool.backward(err)
    print(err.shape)
    print(err[0])

    # err = relu.backward(err)
    # print(err.shape)
    # print(err[0])

    # err = conv2d.backward(err)
    # print(err.shape)
    # print(err[0])

    