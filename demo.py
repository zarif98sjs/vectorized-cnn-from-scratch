import numpy as np
# x = np.ones((3, 3))
# y = np.pad(x, ((1, 2), (1, 0)))
# print(y)

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# strides = (1 * x.strides[0],)
# print(strides)
# y = np.lib.stride_tricks.as_strided(x, shape=(4,), strides=strides)
# print(y)

import numpy as np 
import matplotlib.pyplot as plt

H = np.array([[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]])

## show numbers with labels inside using matplotlib
fig, ax = plt.subplots()
for i in range(4):
    for j in range(4):
        text = ax.text(j, i, H[i, j],
                       ha="center", va="center", color="w")
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=1))
## remove ticks
ax.set_xticks([])
ax.set_yticks([])
## show the matrix with only one color in the background
ax.imshow(H, vmin=12, vmax=12)
# ax.imshow(H, cmap='copper_r')
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(1)
plt.show()