import pickle
import numpy as np

path = "../temp/Result.pickle"

with open(path, "rb") as f:
    res = pickle.load(f)
print(res)


