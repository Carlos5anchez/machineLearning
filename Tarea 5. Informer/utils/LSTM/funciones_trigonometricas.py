import numpy as np


def sigmoid(self, x):
     return 1 / (1 + np.exp(x))

def tanh(self, x):
    return np.exp(x) - np.exp(-x) / (np.exp(x) + np.exp(-x))
     