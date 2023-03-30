import numpy as np

from ias import Iris

x = Iris.features
y = Iris.labels

indices = np.arange(len(x))
np.random.shuffle(indices)
train_size = int(len(x) * 2 / 3)

train_i = indices[:train_size]
train_x = x[train_i]
train_y = y[train_i]

check_i = indices[train_size:]
check_x = x[check_i]
check_y = y[check_i]

# forest.fit(x, y, 250, 50)
