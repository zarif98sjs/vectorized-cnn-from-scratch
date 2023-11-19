"""
CNN from scratch using only numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import os

## set seed for reproducibility
np.random.seed(10)

class Conv2D:
    def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(num_filters)

        self.cache = None

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
        W_col = self.weights.reshape(self.num_filters, -1)
        b_col = self.bias.reshape(-1, 1)

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

        print("dX_col shape: ", dX_col.shape)
        dX = self.col2im(dX_col, x.shape)

        dW = dW_col.reshape(self.num_filters, self.in_channels, self.kernel_size, self.kernel_size)

        return dX, dW, db

class MaxPool2D:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None
        self.method = None

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

        print("X_col shape: ", X_col.shape)

        d_out = d_out.reshape(1, -1)
        print("d_out shape: ", d_out.shape)
        dX_col = np.zeros_like(X_col)
        
        dX_col[max_idx, np.arange(max_idx.size)] = d_out
        print("dX_col shape: ", dX_col.shape)
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

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.cache)

class Dense:
    def __init__(self, out_features, in_features=None):
        self.out_features = out_features
        self.in_features = in_features
        self.W = None
        self.b = None
        self.cache = None
    
    def forward(self, x):
        if self.in_features is None:
            self.in_features = x.shape[1]
            self.W = np.random.randn(self.out_features, self.in_features)
            self.b = np.random.randn(1, self.out_features)
        out = x.dot(self.W.T) + self.b
        self.cache = x
        return out

    def backward(self, d_out):
        x = self.cache
        N = x.shape[0]
        
        dW = d_out.T.dot(x) / N
        db = np.sum(d_out, axis=0, keepdims=True) / N

        dx = d_out.dot(self.W)
        return dx, dW, db

class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        out = np.maximum(0, x)
        self.cache = x
        return out

    def backward(self, d_out):
        # print("d_out shape: ", d_out.shape)
        print(d_out)
        x = self.cache
        d_out[x <= 0] = 0
        return d_out

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, y_pred, y):
        return y_pred - y

# class LeNet:
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.layers = [
#             Conv2D(1, 6, 5, 1, 0),
#             ReLU(),
#             MaxPool2D(2, 2),
#             Conv2D(6, 16, 5, 1, 0),
#             ReLU(),
#             MaxPool2D(2, 2),
#             Flatten(),
#             Dense(120),
#             ReLU(),
#             Dense(84),
#             ReLU(),
#             Dense(self.num_classes)
#         ]

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer.forward(x)
#         return x

#     def backward(self, d_out):
#         for layer in reversed(self.layers):
#             d_out = layer.backward(d_out)
#         return d_out

#     def predict(self, x):
#         x = self.forward(x)
#         return np.argmax(x, axis=1)

#     def evaluate(self, x, y):
#         y_pred = self.predict(x)
#         return np.mean(y_pred == y)

#     def fit(self, x, y, batch_size, epochs, lr, x_val=None, y_val=None):
#         N = x.shape[0]
#         iterations_per_epoch = max(N // batch_size, 1)
#         num_iterations = epochs * iterations_per_epoch

#         for it in range(1, num_iterations + 1):
#             batch_mask = np.random.choice(N, batch_size)
#             x_batch = x[batch_mask]
#             y_batch = y[batch_mask]

#             y_pred = self.forward(x_batch)
#             loss = self.loss(y_pred, y_batch)
#             self.backward(self.gradient(x_batch, y_batch))

#             for layer in self.layers:
#                 if hasattr(layer, 'W'):
#                     layer.W -= lr * layer.dW
#                     layer.b -= lr * layer.db

#             if it % iterations_per_epoch == 0:
#                 train_acc = self.evaluate(x, y)
#                 val_acc = self.evaluate(x_val, y_val) if x_val is not None else 0
#                 print('Epoch %d: loss=%.4f, train_acc=%.3f, val_acc=%.3f' % (it // iterations_per_epoch, loss, train_acc, val_acc))

#     def loss(self, y_pred, y):
#         N = y.shape[0]
#         y_pred = self.softmax(y_pred)
#         return -np.sum(y * np.log(y_pred + 1e-7)) / N
    
#     def softmax(self, x):
#         exps = np.exp(x - np.max(x))
#         return exps / np.sum(exps, axis=1, keepdims=True)

#     def gradient(self, x, y):
#         y_pred = self.forward(x)
#         return self.backward(self.loss_gradient(y_pred, y))

#     def loss_gradient(self, y_pred, y):
#         return (y_pred - y) / y.shape[0]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update(self, layer):
        if self.m is None:
            self.m, self.v = {}, {}
            for p, w in layer.params.items():
                self.m[p] = np.zeros_like(w)
                self.v[p] = np.zeros_like(w)

        self.t += 1
        for p, w in layer.params.items():
            dw = layer.grads[p]
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * dw
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (dw ** 2)

            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            layer.params[p] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class CNNModel():
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, num_filters=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(in_channels=6, num_filters=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(120)
        self.relu3 = ReLU()
        self.fc2 = Dense(84)
        self.relu4 = ReLU()
        self.fc3 = Dense(10)
        self.softmax = Softmax()

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)

        x = self.fc1.forward(x)
        x = self.relu3.forward(x)

        x = self.fc2.forward(x)
        x = self.relu4.forward(x)

        x = self.fc3.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self, y_pred, y):
        delL = self.softmax.backward(y_pred, y)
        delL, dW5, db5 = self.fc3.backward(delL)
        delL = self.relu4.backward(delL)

        delL, dW4, db4 = self.fc2.backward(delL)
        delL = self.relu3.backward(delL)

        delL, dW3, db3 = self.fc1.backward(delL)
        delL = self.flatten.backward(delL)

        delL = self.pool2.backward(delL)
        delL = self.relu2.backward(delL)
        delL, dW2 , db2 = self.conv2.backward(delL)

        delL = self.pool1.backward(delL)
        delL = self.relu1.backward(delL)
        delL , dW1, db1 = self.conv1.backward(delL)

        gradients =  {  'dW1': dW1, 'db1': db1,
                        'dW2': dW2, 'db2': db2,
                        'dW3': dW3, 'db3': db3,
                        'dW4': dW4, 'db4': db4,
                        'dW5': dW5, 'db5': db5  }

        return gradients

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.params['W'] = params['W' + str(i+1)]
            layer.params['b'] = params['b' + str(i+1)]

    def train_step(self, x, y, lr):
        y_pred = self.forward(x)
        loss = self.softmax.loss(y_pred, y)
        gradients = self.backward(y_pred, y)
        self.update_params(gradients, lr)

    def update_params(self, gradients, lr):
        for layer in self.layers:
            for p, w in layer.params.items():
                layer.params[p] -= lr * gradients['d' + p]

    def fit(self, x, y, batch_size=32, epochs=10, lr=0.001):
        for epoch in range(epochs):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.train_step(x_batch, y_batch, lr)

if __name__ == '__main__':

    """
    Test LeNet with dummy data
    """

    x = np.random.randn(100, 1, 28, 28)
    y = np.random.randint(0, 10, (100, 1))

    model = CNNModel()
    model.fit(x, y, batch_size=32, epochs=10, lr=0.001)