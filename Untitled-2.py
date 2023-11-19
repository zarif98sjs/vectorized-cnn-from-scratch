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

    def im2col(self, X, kernel_size, stride, padding):
        N, C, H, W = X.shape
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1

        img = np.pad(X, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

        col = np.zeros((N, C, H_out, W_out, kernel_size, kernel_size))

        for y in range(kernel_size):
            y_max = y + stride * H_out
            for x in range(kernel_size):
                x_max = x + stride * W_out
                col[:, :, :, :, y, x] = img[:, :, y:y_max:stride, x:x_max:stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        return col

    
    def forward(self, x, part1=True, part2=True):

        N, C, H_in, W_in = x.shape
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1

        out1 = None
        ## without using einsum
        if part1:  
            X_col = self.im2col(x, self.kernel_size, self.stride, self.padding)
            W_col = self.weights.reshape(self.num_filters, -1)
            
            out1 = np.dot(X_col, W_col.T) + self.bias.T

            out1 = out1.reshape(self.num_filters, H_out, W_out, N)
            out1 = out1.transpose(3, 0, 1, 2)

        out2 = None
        ## using einsum
        if part2:
            """
            pad over the image dimension only
            reference: https://sparrow.dev/numpy-pad/
            """
            # print("Before padding: ", x.shape)
            x_pad = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
            # print("After padding: ", x_pad.shape)
            ## np.lib.stride_tricks.as_strided
            x_stride = np.lib.stride_tricks.as_strided(x_pad, shape=(N, C, H_out, W_out, self.kernel_size, self.kernel_size), strides=(x_pad.strides[0], x_pad.strides[1], x_pad.strides[2]*self.stride, x_pad.strides[3]*self.stride, x_pad.strides[2], x_pad.strides[3]))

            ## einsum
            # out2 = np.einsum('nchwkm,nkhw->nchw', x_stride, self.weights) + self.bias
            out2 = np.einsum('bihwkl,oikl->bohw', x_stride, self.weights) 
            out2 = out2 + self.bias[None, :, None, None]

        return out1, out2

if __name__ == '__main__':
    conv = Conv2D(3, 8, 3, 1, 1) # in_channels, num_filters, kernel_size, stride, padding
    x = np.random.randn(10, 3, 32, 32) # N, C, H, W
    out1, out2 = conv.forward(x)
    # print(out1 == out2)
    print(out1.shape)
    print(out2.shape)

    ## print numpy array to file
    np.set_printoptions(threshold=np.inf)
    np.savetxt('out1.txt', out1.reshape(-1), fmt='%.6f')
    np.savetxt('out2.txt', out2.reshape(-1), fmt='%.6f')
    