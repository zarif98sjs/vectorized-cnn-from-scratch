class Convolution:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def initialize(self, num_input_features):
        limit = 1 / np.sqrt(num_input_features)
        self.filters = np.random.uniform(-limit, limit, (self.num_filters, num_input_features, self.filter_size, self.filter_size))

    def iterate_regions(self, image):
        h, w = image.shape

        new_h = (h - self.filter_size + 2 * self.padding) // self.stride + 1
        new_w = (w - self.filter_size + 2 * self.padding) // self.stride + 1

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[i * self.stride : i * self.stride + self.filter_size, j * self.stride : j * self.stride + self.filter_size]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learn_rate * d_L_d_filters

        return None
