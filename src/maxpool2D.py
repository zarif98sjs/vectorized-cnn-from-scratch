import numpy as np
np.random.seed(120)

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
        """
        Parameters:
            - dout: Previous layer with the error.
            Returns:
            - dX: Conv layer updated with error.
        """
        if self.method == 'fast':
            return self.backward_fast(d_out)
        elif self.method == 'faster':
            return self.backward_faster(d_out)
        elif self.method == 'slow':
            return self.backward_slow(d_out)
        return None

    def backward_slow(self, dout):
        """
            Distributes error through max pooling layer.

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


    def backward_fast(self, d_out):
        x, X_col, max_idx = self.cache
        N , C, H_in, W_in = x.shape
        d_out = d_out.reshape(1, -1)

        """
        dx_col is the derivative of X_col with respect to the loss.

        We then set the values in dX_col to d_out based on the indices in max_idx, 
        which correspond to the max pooling indices that were calculated in the forward pass
        """
        dX_col = np.zeros_like(X_col)
        dX_col[max_idx, np.arange(max_idx.size)] = d_out
        # print("dX_col shape: ", dX_col.shape)

        """
        col2im here performs the reverse of im2col
        After that we reshape the output to the original shape of x
        """
        dX = self.col2im(dX_col, (N * C, 1, H_in, W_in))
        dX = dX.reshape(x.shape)
        return dX


    def backward_faster(self, d_out):
        """
        Reference:
        https://stackoverflow.com/questions/61954727/max-pooling-backpropagation-using-numpy
        https://gitlab.cs.washington.edu/liangw6/assignment2-for-stanford231n

        "This implementation has a crucial (but often ignored) mistake:
        in case of multiple equal maxima, it backpropagates to all of them which can easily result in
        vanishing / exploding gradients / weights. You can propagate to (any) one of the maximas, not all of them.
        tensorflow chooses the first maxima."

        """
        """
        In this backward pass, the gradient with respect to the input x is computed, given the gradient with respect to the output d_out.
        For this efficient implementation, in the forward pass, the input x is divided into non-overlapping regions and the maximum value in each region is selected as the output.
        And in the backward pass, the gradient is propagated back to the input x in a way that only the elements of x corresponding to the maximum value in each region receive the gradient.
        """
        x, x_split, out = self.cache
        dx_split = np.zeros_like(x_split)
        """
        The purpose of mask is to identify which elements of x correspond to the maximum value in each region, and then distribute the gradient to these elements.
        """
        mask = (x_split == np.expand_dims(np.expand_dims(out, 3), 5))
        dout_broadcast, _ = np.broadcast_arrays(np.expand_dims(np.expand_dims(d_out, 3), 5), dx_split)
        """
        This mask is then used to update the values of dx_split.
        The values of dx_split are updated by taking d_out of the mask (after boradcasting), and then normalizing by the sum of the mask along the 3rd and 5th dimensions.
        The 3rd and 5th dimensions are from this shape: (N, C, H_out, F, W_out, F)
        """
        dx_split[mask] = dout_broadcast[mask]
        dx_split /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_split.reshape(x.shape)
        return dx
