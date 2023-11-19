import numpy as np

class Softmax:
    def __init__(self):
        self.trainable = False
        pass

    def __str__(self):
        return "Softmax"

    def forward(self, x):
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, d_out):
        return d_out