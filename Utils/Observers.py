import numpy as np


def empty_observer(path):
    return []


def full_observer(path):
    return path


def mean_cov(path):
    p = np.array(path).T
    return p.mean(axis=1) + np.diag(np.cov(p))
