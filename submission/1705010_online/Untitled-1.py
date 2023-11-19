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

    def im2col(self, X):
        img = np.pad(X, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        i, j, d = self.get_matrix_indices(X.shape)
        cols = img[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def col2im(self, X):
        _ , C, H, W = X.shape
        
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

    def backward(self, dout):
        x, X_col, W_col = self.cache

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(self.num_filters, -1)

        dW = dout @ X_col.T
        dW = dW.reshape(self.weights.shape)

        dX_col = W_col.T @ dout
        dX_col = dX_col.reshape(x.shape[1], self.kernel_size, self.kernel_size, x.shape[2], x.shape[3])
        dX_col = np.moveaxis(dX_col, 0, -1)
        dX_col = np.moveaxis(dX_col, 0, -1)
        dX_col = np.moveaxis(dX_col, 0, -1)

        return dX_col, dW, db

if __name__ == '__main__':
    conv = Conv2D(3, 8, 3, 1, 1) # in_channels, num_filters, kernel_size, stride, padding
    x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    out1, out2 = conv.forward(x)
    # check if all the values are equal
    print(np.allclose(out1, out2))
    print(out1.shape)
    print(out2.shape)

    ## print numpy array to file
    np.set_printoptions(threshold=np.inf)
    np.savetxt('out1.txt', out1.reshape(-1), fmt='%.6f')
    np.savetxt('out2.txt', out2.reshape(-1), fmt='%.6f')
    