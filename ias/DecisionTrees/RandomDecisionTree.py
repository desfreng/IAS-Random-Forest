import numpy as np
from graphviz import Digraph

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_mean_criterion, criterion, subset_bagging, calculate_gini, calculate_log_loss, random


class RandomDecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf, subset_size: int | None = None,
                 criterion_name: str = "gini"):
        super().__init__(max_depth, "gini")
        self._subset_size = subset_size
        self.compute_criterion = calculate_gini if criterion_name == "gini" else calculate_log_loss

    @staticmethod
    def _threshold_array(self, attr_bag):
        return random(np.min(attr_bag), np.max(attr_bag))

    def _find_threshold(self, data_set, label_set) -> tuple[criterion, int, float]:
        """
        Chooses best threshold to split the dataset between random threshold for each feature.
        Random subspaces is applied to y.
        :param data_set: the dataset
        :param label_set: the label_set
        :return: tuple containing (the best criterion value, feature number, threshold value)
        """
        best_criterion = 2

        if self._subset_size is None:
            bag = subset_bagging(int(np.sqrt(self.features_number)), self.features_number)
        else:
            bag = subset_bagging(self._subset_size, self.features_number)

        right_labels, left_labels = np.empty(0), np.empty(0)
        thresholds = np.apply_along_axis(self._threshold_array, 1, np.transpose(data_set[:, bag])[bag])
        res_f, res_t = 0, thresholds[0]
        for f, t in enumerate(thresholds):  # on itère sur les thresholds tirés au hasard
            left_labels = label_set[np.argwhere(data_set[:, f] < t), :]
            right_labels = label_set[np.argwhere(data_set[:, f] >= t), :]
            new_crit = calculate_mean_criterion(left_labels, right_labels, self.compute_criterion)
            if new_crit < best_criterion:
                best_criterion = new_crit
                res_f, res_t = f, t

        if right_labels.size == 0 or left_labels.size == 0:
            return self._find_threshold(data_set, label_set)

        return best_criterion, res_f, res_t

    def show(self, features_names=None, class_name=None) -> Digraph:
        return self._abstract_show(True, features_names, class_name)
