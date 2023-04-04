import numpy as np

from ias import NaiveDecisionTree, PCA
from ias.Datasets import Emnist
from ias.Metrics import accuracy_score, confusion_matrix, show_confusion_matrix

print("Start")
indices = np.arange(len(Emnist.attributes))
np.random.shuffle(indices)
train_size = 1000
test_size = 1000

train_i = indices[:train_size]
train_x = Emnist.attributes[train_i]
train_y = Emnist.labels[train_i]

test_i = indices[train_size:]
test_x = Emnist.attributes[test_i]
test_y = Emnist.labels[test_i]

print("PCA")

pca = PCA(0.8)
pca.fit(train_x)

print("Compression")
compressed_train_x = pca.compress(train_x)
compressed_test_x = pca.compress(test_x)

print("Tree Fit")
tree = NaiveDecisionTree(max_depth=10)
tree.fit(compressed_train_x, train_y)

print("Results")
fig, ax = show_confusion_matrix(confusion_matrix(Emnist.class_number, test_y,
                                                 tree.predict(compressed_test_x)),
                                Emnist.class_names)
fig.show()

print(accuracy_score(test_y, tree.predict(compressed_test_x)))
