import pickle
import numpy as np
import matplotlib.pyplot as plt

class History:
    def __init__(self):
        self.history = {'loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    def add(self, loss, val_loss, val_acc, val_f1):
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)

    def plot(self):
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['loss'], label='train_loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig('loss.png')
        plt.clf()

        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig('acc.png')
        plt.clf()

        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['val_f1'], label='val_f1')
        plt.legend()
        plt.savefig('f1.png')
        plt.clf()

        """
        plot all in same fig
        """

        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(self.history['loss'], label='train_loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.plot(np.array(self.history['val_acc'])/100, label='val_acc')
        plt.plot(np.array(self.history['val_f1'])/100, label='val_f1')
        plt.legend()
        plt.savefig("all_metric_in_one.png")
        plt.clf()

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.history, f)