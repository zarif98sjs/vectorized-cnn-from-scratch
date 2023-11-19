import numpy as np
np.random.seed(120)

class Dense:
    def __init__(self, out_features, in_features=None):
        self.out_features = out_features
        self.in_features = in_features
        W = None
        b = None

        self.trainable = True
        self.W = {"val": W, "grad": 0}
        self.b = {"val": b, "grad": 0}

        self.cache = None

    def __str__(self):
        return "Dense({})".format(self.out_features)

    def forward(self, x):
        if self.W["val"] is None:
            self.in_features = x.shape[1]
            self.W["val"] = np.random.randn(self.out_features, self.in_features) * np.sqrt(1.0 / self.in_features)
            self.b["val"] = np.random.randn(1, self.out_features) * np.sqrt(1.0 / self.in_features)
        out = x.dot(self.W["val"].T) + self.b["val"]
        self.cache = x
        return out

    def backward(self, d_out):
        x = self.cache
        N = x.shape[0]

        dW = d_out.T.dot(x) / N
        db = np.sum(d_out, axis=0, keepdims=True) / N

        dx = d_out.dot(self.W["val"])

        self.W["grad"] = dW
        self.b["grad"] = db

        return dx
