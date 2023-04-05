import numpy as np

from ias import NaiveDecisionTree
from ias.Datasets import Iris
from ias.Metrics import accuracy_score, confusion_matrix, show_confusion_matrix

x = Iris.attributes
y = Iris.labels

print("Start")
indices = np.arange(len(x))
np.random.shuffle(indices)
train_size = int(len(Iris.attributes) * 2 / 3)

train_i = indices[:train_size]
train_x = x[train_i]
train_y = y[train_i]

test_i = indices[train_size:]
test_x = x[test_i]
test_y = y[test_i]

print("PCA")

# pca = PCA(0.8)
# pca.fit(train_x)
#
# print("Compression")
# compressed_train_x = pca.compress(train_x)
# compressed_test_x = pca.compress(test_x)

print("Tree Fit")
tree = NaiveDecisionTree()
tree.fit(train_x, train_y)

print("Results")
fig, ax = show_confusion_matrix(confusion_matrix(Iris.class_number, test_y,
                                                 tree.predict(test_x)),
                                Iris.class_names)
fig.show()

print(accuracy_score(test_y, tree.predict(test_x)))
