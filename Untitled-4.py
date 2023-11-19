class FullyConnectedLayer():
    def __init__(self, in_size, out_size, std=1e-4):
        self.params = {}
        self.params['W'] = std * np.random.randn(in_size, out_size)
        self.params['b'] = np.zeros(out_size)
        self.grads = {}
        self.cache = None

    def forward(self, x):
        W, b = self.params['W'], self.params['b']
        out = np.dot(x, W) + b
        self.cache = x
        return out

    def backward(self, d_out):
        x = self.cache
        W, b = self.params['W'], self.params['b']

        dx = np.dot(d_out, W.T)
        dW = np.dot(x.T, d_out)
        db = np.sum(d_out, axis=0)

        self.grads['W'] = dW
        self.grads['b'] = db

        return dx