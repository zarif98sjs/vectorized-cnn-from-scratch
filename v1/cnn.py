"""
CNN from scratch using only numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def im2col(X, kernel_h, kernel_w, stride=1, padding=0):
    N, C, H, W = X.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    img = np.pad(X, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, N, C, H, W, kernel_h, kernel_w, stride=1, padding=0):
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, padding:H + padding, padding:W + padding]

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.random.randn(out_channels)
        self.X = None
        self.X_col = None
        self.W_col = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        X_col = im2col(X, self.kernel_size, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.out_channels, -1)
        out = np.dot(W_col, X_col) + self.b.reshape(-1, 1)
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        N, C, H, W = self.X.shape
        dout = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.db = np.sum(dout, axis=1)
        self.dW = np.dot(dout, self.X_col.T).reshape(self.W.shape)
        dX_col = np.dot(self.W_col.T, dout)
        dX = col2im(dX_col, N, C, H, W, self.kernel_size, self.kernel_size, self.stride, self.padding)
        return dX

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class MaxPool2d:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.X = None
        self.X_col = None
        self.arg_max = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        X_col = im2col(X, self.kernel_size, self.kernel_size, self.stride, self.padding)
        X_col = X_col.reshape(-1, self.kernel_size * self.kernel_size)
        arg_max = np.argmax(X_col, axis=1)
        out = np.max(X_col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.kernel_size * self.kernel_size
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dX_col = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dX = col2im(dX_col, dout.shape[0], dout.shape[1], dout.shape[2], dout.shape[3], self.kernel_size, self.kernel_size, self.stride, self.padding)
        return dX

class Flatten:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out = X.reshape(N, -1)
        return out

    def backward(self, dout):
        N, C, H, W = self.X.shape
        dX = dout.reshape(N, C, H, W)
        return dX

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features)
        self.b = np.random.randn(out_features)
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        out = np.dot(X, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        dX = np.dot(dout, self.W.T)
        return dX

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dX = dout
        return dX

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * (1.0 - self.out) * self.out
        return dX

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx

class Model:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, X, t):
        y = self.predict(X)
        return cross_entropy_error(y, t)

    def accuracy(self, X, t):
        y = self.predict(X)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, t):
        loss_W = lambda W: self.loss(X, t)
        grads = {}
        for idx in range(1, len(self.layers)+1):
            layer = self.layers[idx-1]
            if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                grads['W' + str(idx)] = numerical_gradient(loss_W, layer.W)
                grads['b' + str(idx)] = numerical_gradient(loss_W, layer.b)
        return grads

    def gradient(self, X, t):
        self.loss(X, t)
        dout = 1
        dout = self.layers[-1].backward(dout)
        for layer in reversed(self.layers[:-1]):
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, len(self.layers)+1):
            layer = self.layers[idx-1]
            if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                grads['W' + str(idx)] = layer.dW
                grads['b' + str(idx)] = layer.db
        return grads

    def update(self, lr):
        for layer in self.layers:
            if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                layer.update(lr)

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def fit(self, X, t, epochs=10, batch_size=32, verbose=1):
        data_size = X.shape[0]
        max_iter = data_size // batch_size
        for epoch in range(epochs):
            idx = np.random.permutation(data_size)
            X = X[idx]
            t = t[idx]
            for it in range(max_iter):
                X_batch = X[it*batch_size:(it+1)*batch_size]
                t_batch = t[it*batch_size:(it+1)*batch_size]
                grads = self.model.gradient(X_batch, t_batch)
                self.optimizer.update(self.model.layers, grads)
            loss = self.model.loss(X_batch, t_batch)
            if verbose:
                print('epoch %d, loss %.4f' % (epoch+1, loss))

    def evaluate(self, X, t):
        loss = self.model.loss(X, t)
        acc = self.model.accuracy(X, t)
        print('loss %.4f, acc %.4f' % (loss, acc))

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layers, grads):
        for idx in range(1, len(layers)+1):
            layer = layers[idx-1]
            if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                layer.W -= self.lr * grads['W' + str(idx)]
                layer.b -= self.lr * grads['b' + str(idx)]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, layers, grads):
        if self.v is None:
            self.v = {}
            for idx in range(1, len(layers)+1):
                layer = layers[idx-1]
                if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                    self.v['W' + str(idx)] = np.zeros_like(layer.W)
                    self.v['b' + str(idx)] = np.zeros_like(layer.b)
        for idx in range(1, len(layers)+1):
            layer = layers[idx-1]
            if isinstance(layer, Conv2d) or isinstance(layer, Linear):
                self.v['W' + str(idx)] = self.momentum * self.v['W' + str(idx)] - self.lr * grads['W' + str(idx)]
                self.v['b' + str(idx)] = self.momentum * self.v['b' + str(idx)] - self.lr * grads['b' + str(idx)]
                layer.W += self.v['W' + str(idx)]
                layer.b += self.v['b' + str(idx)]

