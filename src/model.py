import json
import pickle
from src.relu import ReLU
from src.dense import Dense
from src.conv2D import Conv2D
from src.flatten import Flatten
from src.softmax import Softmax
from src.maxpool2D import MaxPool2D

class CNNModel():
    def __init__(self):
        self.layers = []
        self.load_layer_from_json("model.json")

    def __str__(self):
        ret = ""
        for layer in self.layers:
            ret += str(layer) + "\n"
        return ret

    def load_layer_from_json(self, file_name):
        with open (file_name, "r") as f:
            layer_params = json.load(f)

        for layer in layer_params["model"]:
            if "Conv2D" in layer.keys():
                self.layers.append(Conv2D(layer["Conv2D"]["in_channels"], layer["Conv2D"]["num_filters"], layer["Conv2D"]["kernel_size"], layer["Conv2D"]["stride"], layer["Conv2D"]["padding"]))
            elif "ReLU" in layer.keys():
                self.layers.append(ReLU())
            elif "MaxPool2D" in layer.keys():
                self.layers.append(MaxPool2D(layer["MaxPool2D"]["kernel_size"], layer["MaxPool2D"]["stride"]))
            elif "Flatten" in layer.keys():
                self.layers.append(Flatten())
            elif "Dense" in layer.keys():
                self.layers.append(Dense(layer["Dense"]["out_features"]))
            elif "Softmax" in layer.keys():
                self.layers.append(Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, delL):
        for layer in reversed(self.layers):
            delL = layer.backward(delL)

        gradients = {}
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                gradients['dW' + str(i+1)] = layer.W['grad']
                gradients['db' + str(i+1)] = layer.b['grad']
        return gradients

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                params['W' + str(i+1)] = layer.W['val']
                params['b' + str(i+1)] = layer.b['val']
        return params

    def set_params(self, params, lr):
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer.W['val'] -= lr * params['dW'+ str(i+1)]
                layer.b['val'] -= lr * params['db'+ str(i+1)]

    def save_model_weights_pickle(self, file_name):
        params = self.get_params()
        with open(file_name, "wb") as f:
            pickle.dump(params, f)


    def load_model_weights_pickle(self, file_name):
        with open(file_name, "rb") as f:
            params_new = pickle.load(f)

        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer.W['val'] = params_new['W' + str(i+1)]
                layer.b['val'] = params_new['b' + str(i+1)]