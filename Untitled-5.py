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
        x = self.cache
        dx = d_out.copy()
        dx[x <= 0] = 0
        return dx

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, y_pred, y):
        return y_pred - y

class LeNet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = [
            Conv2D(1, 6, 5, 1, 0),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(6, 16, 5, 1, 0),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(120),
            ReLU(),
            Dense(84),
            ReLU(),
            Dense(self.num_classes)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def predict(self, x):
        x = self.forward(x)
        return np.argmax(x, axis=1)

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)

    def fit(self, x, y, batch_size, epochs, lr, x_val=None, y_val=None):
        N = x.shape[0]
        iterations_per_epoch = max(N // batch_size, 1)
        num_iterations = epochs * iterations_per_epoch

        for it in range(1, num_iterations + 1):
            batch_mask = np.random.choice(N, batch_size)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]

            y_pred = self.forward(x_batch)
            loss = self.loss(y_pred, y_batch)
            self.backward(self.gradient(x_batch, y_batch))

            for layer in self.layers:
                if hasattr(layer, 'W'):
                    layer.W -= lr * layer.dW
                    layer.b -= lr * layer.db

            if it % iterations_per_epoch == 0:
                train_acc = self.evaluate(x, y)
                val_acc = self.evaluate(x_val, y_val) if x_val is not None else 0
                print('Epoch %d: loss=%.4f, train_acc=%.3f, val_acc=%.3f' % (it // iterations_per_epoch, loss, train_acc, val_acc))

    def loss(self, y_pred, y):
        N = y.shape[0]
        y_pred = self.softmax(y_pred)
        return -np.sum(y * np.log(y_pred + 1e-7)) / N
    

if __name__ == '__main__':

    """
    Test LeNet with dummy data
    """

    

    # conv = Conv2D(3, 8, 3, 1, 1) # in_channels, num_filters, kernel_size, stride, padding
    # x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    # out = conv.forward(x)
    # print(out.shape)

    # ## test backward
    # dout = np.random.randn(10, 8, 32, 32)
    # dx, dw, db = conv.backward(dout)
    # print(dx.shape, dw.shape, db.shape)

    # ## print numpy array to file
    # np.set_printoptions(threshold=np.inf)
    # np.savetxt('out_FAST_v1_fw.txt', out.reshape(-1), fmt='%.6f')

    # maxpool = MaxPool2D(2, 2)
    # # x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    # ## generate random input integer
    # # x = np.random.randint(0, 100, size=(1, 2, 4, 4))
    # # x = np.random.randn(1, 1, 8, 8) # N, C, H, W
    # x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    # # print(x)
    # out = maxpool.forward_fast(x)
    # # print(out)
    # print(out.shape)
    

    # np.set_printoptions(threshold=np.inf)
    # np.savetxt('maxpool_FAST_v1_fwfast_out.txt', out.reshape(-1), fmt='%.6f')

    """
    Test Conv2D and MaxPool2D
    """

    # conv = Conv2D(3, 8, 3, 1, 1) # in_channels, num_filters, kernel_size, stride, padding
    # x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    # out = conv.forward(x)
    # print(out.shape)

    # ## test backward
    # dout = np.random.randn(10, 8, 32, 32)
    # dx, dw, db = conv.backward(dout)
    # print(dx.shape, dw.shape, db.shape)

    # print("------------------")
    
    # maxpool = MaxPool2D(2, 2)
    # ## test forward
    # x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    # out = maxpool.forward_faster(x)
    # print(out.shape)

    # ## test backward
    # dout = np.random.randn(10, 3, 16, 16)
    # dx = maxpool.backward_faster(dout)
    # print(dx.shape)


    # np.set_printoptions(threshold=np.inf)
    # np.savetxt('maxpool_FAST_v1_bwfaster_out.txt', dx.reshape(-1), fmt='%.6f')

    """
    Test Dense layer
    """

    # dense = Dense(out_features=10)
    # x = np.random.randn(10, 5)
    # out = dense.forward(x)
    # print(out.shape)

    # dout = np.random.randn(10, 10)
    # dx, dw, db = dense.backward(dout)
    # print(dx.shape, dw.shape, db.shape)

    """
    Test Flatten layer
    """

    # flatten = Flatten()
    # x = np.random.randn(10, 3, 32, 32)
    # out = flatten.forward(x)
    # print(out.shape)

    # dout = np.random.randn(10, 3*32*32)
    # dx = flatten.backward(dout)
    # print(dx.shape)