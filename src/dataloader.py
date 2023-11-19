import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, label_paths):
        self.label_paths = label_paths

    def load(self, isTestSet=False):
        df = pd.DataFrame()
        for label_path in self.label_paths:
            df = df.append(pd.read_csv(label_path))
        COUNT = 100
        if isTestSet:
            COUNT = 50000

        labels = df['digit'].values

        filenames = df['database name'].values + '/' + df['filename'].values
        labels = labels[:COUNT]
        filenames = filenames[:COUNT]

        images = []
        for filename in filenames:
            image = cv2.imread('dataset/NumtaDB_with_aug/' + filename, cv2.IMREAD_COLOR)
            images.append(image)

        ## resize
        images = [cv2.resize(image, (28, 28)) for image in images]
        images = np.array(images)
        print(images.shape)

        ## adjust dimensions : (N, H, W, C) -> (N, C, H, W)
        images = np.transpose(images, (0, 3, 1, 2))

        ## normalize
        images = images / 255.0

        ## convert (N, C, H, W) -> (N, 1, H, W)
        # images = images[:, 0:1, :, :]
        # print(images.shape)

        ## change background color
        images = 1 - images

        if isTestSet:
            labels = labels.reshape(-1, 1)
            return images, labels


        """
        stratified split
        """

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        """
        print sample distribution
        """

        print("Train distribution")
        print(np.unique(y_train, return_counts=True))
        print("Test distribution")
        print(np.unique(y_test, return_counts=True))

        return X_train, X_test, y_train, y_test
