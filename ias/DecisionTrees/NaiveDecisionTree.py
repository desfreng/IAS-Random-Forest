import numpy as np
from graphviz import Digraph

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_mean_criterion, shrunk_proba_vector


class NaiveDecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf):
        super().__init__(max_depth, "gini")

    def _find_threshold(self, data_set, label_set) -> tuple[float, int, int]:
        """ Finds best threshold to split the dataset. """
        best_gini = None
        best_feature = None
        best_threshold = None

        for feature in range(self.features_number):
            feature_data = data_set[:, feature].flatten()

            for threshold in feature_data:
                left_indexes = np.argwhere(feature_data <= threshold)
                right_indexes = np.argwhere(feature_data > threshold)

                gini = calculate_mean_criterion(label_set[left_indexes], label_set[right_indexes],
                                                self.compute_criterion)
                if best_gini is None or gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_gini, best_feature, best_threshold

    def compute_criterion(self, label_set: np.ndarray) -> float:
        """
        calculate gini index from array of class labels
        """
        return 1 - np.sum(shrunk_proba_vector(label_list=label_set) ** 2)

    def show(self, features_names=None, class_name=None) -> Digraph:
        return self._abstract_show(True, features_names, class_name)
