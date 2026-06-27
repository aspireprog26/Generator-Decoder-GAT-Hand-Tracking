import numpy as np 

def ema(self, arr, alpha, axis):
    arr = np.asarray(arr)
    x = np.moveaxis(arr, axis, 0)

    y = np.empty_like(x, dtype = float)
    y[0] = x[0]

    for i in range(1, x.shape[0]):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]

    return np.moveaxis(y, 0, axis)

def transformImage()