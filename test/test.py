import numpy as np

from ias.Datasets import Iris
from ias.DecisionTrees.CartDecisionTree import CartDecisionTree

indices = np.arange(len(Iris.attributes))
np.random.shuffle(indices)
train_size = int(len(Iris.attributes) * 2 / 3)

train_i = indices[:train_size]
train_x = Iris.attributes[train_i]
train_y = Iris.labels[train_i]

test_i = indices[train_size:]
test_x = Iris.attributes[test_i]
test_y = Iris.labels[test_i]

t = CartDecisionTree()
t.fit(train_x, train_y)
t.show(Iris.features_names, Iris.class_names)
t.predict_proba(test_x)
