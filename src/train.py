import time
from utils import train, test
from dataloader import DataLoader

if __name__ == '__main__':

    start = time.time()
    dataloader = DataLoader(
        label_paths = [
            'dataset/NumtaDB_with_aug/training-a.csv',
            'dataset/NumtaDB_with_aug/training-b.csv',
            'dataset/NumtaDB_with_aug/training-c.csv'
        ]
    )
    x, val_x, y, val_y = dataloader.load()
    print("Time taken to load data: ", time.time() - start)

    train(x, y, val_x, val_y)
    test(val_x, val_y)