import numpy as np

from ias.Datasets import Iris

x = Iris.features
y = Iris.labels

indices = np.arange(len(x))
np.random.shuffle(indices)
train_size = int(len(x) * 2 / 3)

train_i = indices[:train_size]
train_x = x[train_i]
train_y = y[train_i]

test_i = indices[train_size:]
test_x = x[test_i]
test_y = y[test_i]

