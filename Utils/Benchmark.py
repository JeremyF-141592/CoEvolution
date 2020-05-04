import numpy as np


def gramacy_lee(x):
    res = (x-1)**4 + np.sin(10*np.pi*x)/(2*x)
    maximum = 0.548563444114526
    return -res, maximum
