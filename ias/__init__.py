from .Datasets import Dataset
from .DecisionTree import DecisionTree
from .PCA import PCA
from .RandomForest import RandomForest

Enmist = Dataset("enmist")
Iris = Dataset("iris")

__all__ = ["PCA", "DecisionTree", "RandomForest", "Enmist", "Iris"]
