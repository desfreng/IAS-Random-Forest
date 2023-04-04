import numpy as np
from graphviz import Digraph

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_gini, calculate_log_loss, calculate_mean_criterion, class_id, \
    criterion, subset_bagging


class CartDecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf, subset_size: int | None = None,
                 criterion_name: str = "gini"):
        super().__init__(max_depth, "gini")
        self._subset_size = subset_size
        self._criterion_name = criterion_name

    def compute_criterion(self, label_set: np.ndarray[class_id]) -> criterion:
        if self._criterion_name == "gini":
            return calculate_gini(label_set)
        else:
            return calculate_log_loss(label_set)

    def _find_threshold(self, data_set, label_set) -> tuple[criterion, int, float]:
        """
        Finds best threshold to split the dataset.
        The best (feature, threshold) is chosen between a random subset of y.
        :param data_set: the dataset
        :param label_set: the label_set
        :return: tuple containing (the best criterion value, feature number, threshold value)
        """
        best_criterion = None
        best_feature = None
        best_threshold = None

        if self._subset_size is None:
            bag = subset_bagging(int(np.sqrt(self.features_number)), self.features_number)
        else:
            bag = subset_bagging(self._subset_size, self.features_number)

        for feature in bag:
            feature_data = data_set[:, feature].flatten()
            for threshold in feature_data:
                left_indexes = np.argwhere(feature_data <= threshold)
                right_indexes = np.argwhere(feature_data > threshold)

                current_criterion = calculate_mean_criterion(label_set[left_indexes],
                                                             label_set[right_indexes],
                                                             self.compute_criterion)
                
                if (best_criterion is None or current_criterion < best_criterion) \
                        and len(left_indexes) > 0 \
                        and len(right_indexes) > 0:
                    best_criterion = current_criterion
                    best_feature = feature
                    best_threshold = threshold

            if best_criterion is None:
                return self._find_threshold(data_set, label_set)

        return best_criterion, best_feature, best_threshold

    def show(self, features_names=None, class_name=None) -> Digraph:
        return self._abstract_show(True, features_names, class_name)
