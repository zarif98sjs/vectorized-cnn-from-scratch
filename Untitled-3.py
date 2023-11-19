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

    def get_indices(self, X_shape, HF, WF, stride, pad):
        """
            Returns index matrices in order to transform our input image into a matrix.

            Parameters:
            -X_shape: Input image shape.
            -HF: filter height.
            -WF: filter width.
            -stride: stride value.
            -pad: padding value.

            Returns:
            -i: matrix of index i.
            -j: matrix of index j.
            -d: matrix of index d. 
                (Use to mark delimitation for each channel
                during multi-dimensional arrays indexing).
        """
        # get input size
        m, n_C, n_H, n_W = X_shape

        # get output size
        out_h = int((n_H + 2 * pad - HF) // stride) + 1
        out_w = int((n_W + 2 * pad - WF) // stride) + 1
    
        # ----Compute matrix of index i----

        # Level 1 vector.
        level1 = np.repeat(np.arange(HF), WF)
        # Duplicate for the other channels.
        level1 = np.tile(level1, n_C)
        # Create a vector with an increase by 1 at each level.
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        # Create matrix of index i at every levels for each channel.
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        # ----Compute matrix of index j----
        
        # Slide 1 vector.
        slide1 = np.tile(np.arange(WF), HF)
        # Duplicate for the other channels.
        slide1 = np.tile(slide1, n_C)
        # Create a vector with an increase by 1 at each slide.
        everySlides = stride * np.tile(np.arange(out_w), out_h)
        # Create matrix of index j at every slides for each channel.
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

        # ----Compute matrix of index d----

        # This is to mark delimitation for each channel
        # during multi-dimensional arrays indexing.
        d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

        return i, j, d


    def im2col(self, X, kernel_size, stride, padding):
        N, C, H, W = X.shape
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1

        img = np.pad(X, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
        i, j, d = self.get_indices(X.shape, kernel_size, kernel_size, stride, padding)
        cols = img[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

        # col = np.zeros((N, C, H_out, W_out, kernel_size, kernel_size))

        # ## consider padding when converting to column
        # for i in range(kernel_size):
        #     i_max = i + stride * H_out
        #     for j in range(kernel_size):
        #         j_max = j + stride * W_out
        #         col[:, :, :, :, i, j] = img[:, :, i:i_max:stride, j:j_max:stride]
        
        # col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        # return col
    
    def forward(self, x, part1=True, part2=True):

        N, C, H_in, W_in = x.shape
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1

        out1 = None
        ## without using einsum
        if part1:  
            X_col = self.im2col(x, self.kernel_size, self.stride, self.padding)
            W_col = self.weights.reshape(self.num_filters, -1)
            b_col = self.bias.reshape(-1, 1)

            out1 = W_col @ X_col + b_col 

            out1 = np.array(np.hsplit(out1, N)).reshape(N, self.num_filters, H_out, W_out)

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
    