"""
sum these values:
8.91279366e+10 -1.55250007e+10 -2.34327009e+10 -1.05358457e+10
  -6.82298647e+09  3.83011841e+09 -3.52461209e+09 -3.86603788e+09
  -1.83993479e+10 -1.08515233e+10


8.47461235e+10 -1.47617397e+10 -2.22806748e+10 -1.00178687e+10
  -6.48754461e+09  3.64181836e+09 -3.35132998e+09 -3.67597119e+09
  -1.74947839e+10 -1.03180290e+10
"""

import numpy as np
import os

# v = np.array([9.48203417e-01, 3.67792526e-03, 4.08020915e-08, 8.01349422e-10, 1.57418807e-11, 4.80832186e-02, 3.26266275e-05, 8.15090025e-09, 5.77777183e-07, 2.18523787e-06])
v1 = np.array([8.91279366e+10, -1.55250007e+10, -2.34327009e+10, -1.05358457e+10, -6.82298647e+09, 3.83011841e+09, -3.52461209e+09, -3.86603788e+09, -1.83993479e+10, -1.08515233e+10])
v2 = np.array([8.47461235e+10, -1.47617397e+10, -2.22806748e+10, -1.00178687e+10, -6.48754461e+09, 3.64181836e+09, -3.35132998e+09, -3.67597119e+09, -1.74947839e+10, -1.03180290e+10])
v = np.array([v1, v2])
print(v)
print(v.shape)

## calc softmax
# def softmax(x):
#     exp = np.exp(x - np.max(x))
#     return exp / np.sum(exp, axis=1)[:, np.newaxis]

def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax2(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


print(softmax2(v1))
print(softmax2(v2))

print(softmax(v))