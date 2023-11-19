class Flatten:
    def __init__(self):
        self.cache = None
        self.trainable = False

    def __str__(self):
        return "Flatten"

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.cache)
