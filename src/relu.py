import numpy as np

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
        d_out[x <= 0] = 0
        return d_out
