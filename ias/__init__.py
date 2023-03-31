from .Datasets import Dataset
from .DecisionTree import DecisionTree
from .PCA import PCA
from .RandomForest import RandomForest

Emnist = Dataset("emnist")
Iris = Dataset("iris")

__all__ = ["PCA", "DecisionTree", "RandomForest", "Emnist", "Iris"]
