import sys
import time
from utils import predict
from dataloader import TestDataLoader

if __name__ == '__main__':

    """
    TEST SET: training-d.csv
    """

    start = time.time()
    testdataloader = TestDataLoader(sys.argv[1])
    test_x, image_names = testdataloader.load()
    print("Time taken to load data: ", time.time() - start)

    predict(test_x, image_names)