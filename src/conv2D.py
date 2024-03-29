import numpy as np
np.random.seed(120)

class Conv2D:
    def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weights = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * np.sqrt(1. / (self.kernel_size))
        bias = np.zeros(num_filters) * np.sqrt(1. / (self.kernel_size))

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
        """
        calculate the output dimensions using the formula:
        out = (in + 2*padding - kernel_size) // stride + 1
        """
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1

        """
        convert image to columns for purpose of vectorization
        """
        X_col = self.im2col(x)
        W_col = self.W["val"].reshape(self.num_filters, -1)
        b_col = self.b["val"].reshape(-1, 1)

        """
        convolution can now be performed as matrix multiplication
        """
        out = W_col @ X_col + b_col

        """
        the convolution output for each sample is stacked side by side
        we reshape it and make it stack vertically/ top to bottom
        """
        out = np.array(np.hsplit(out, N))
        out = out.reshape(N, self.num_filters, H_out, W_out)

        self.cache = (x, X_col, W_col)
        return out

    def backward(self, d_out):
        print("d_out : ", d_out.shape)
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

        dX = self.col2im(dX_col, x.shape)
        dW = dW_col.reshape(self.num_filters, self.in_channels, self.kernel_size, self.kernel_size)

        self.W["grad"] = dW
        self.b["grad"] = db

        return dX
