from .Datasets import Dataset, install_emnist, install_iris
from .DecisionTree import DecisionTree
from .PCA import PCA
from .RandomForest import RandomForest

Enmist = Dataset("enmist")
Iris = Dataset("iris")

__all__ = ["PCA", "DecisionTree", "RandomForest", "Enmist", "Iris", "install_emnist",
           "install_iris"]
