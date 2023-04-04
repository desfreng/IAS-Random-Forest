import numpy as np
from graphviz import Digraph

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_mean_criterion, criterion, subset_bagging, calculate_gini, calculate_log_loss


class DecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf, subset_size: int | None = None,
                 criterion_name: str = "gini"):
        super().__init__(max_depth, "gini")
        self._subset_size = subset_size
        self.compute_criterion = calculate_gini if criterion_name == "gini" else calculate_log_loss

    def _find_threshold(self, data_set, label_set) -> tuple[criterion, int, float]:
        """
        Finds best threshold to split the dataset.
        The best (feature, threshold) is chosen between a random subset of y.
        :param data_set: the dataset
        :param label_set: the label_set
        :return: tuple containing (the best criterion value, feature number, threshold value)
        """
        best_criterion = 2

        if self._subset_size is None:
            bag = subset_bagging(int(np.sqrt(self.features_number)), self.features_number)
        else:
            bag = subset_bagging(self._subset_size, self.features_number)

        res_f, res_t = bag[0], data_set[0][bag[0]]
        r_x = np.empty(0)
        for f in bag:  # on itère sur les features d'un random subset
            l_x, r_x = np.empty(data_set.shape), data_set[data_set[:, f].argsort()]
            l_y, r_y = np.empty(label_set.shape), label_set[data_set[:, f].argsort()]
            for i, e in enumerate(data_set):
                np.add(l_x, e)
                r_x = r_x[i + 1:]
                new_crit = calculate_mean_criterion(l_y, r_y, self.compute_criterion)
                if new_crit < best_criterion:
                    best_criterion = new_crit
                    res_f, res_t = f, r_x[0][f]
        if r_x.size == 0:
            return self._find_threshold(data_set, label_set)

        return best_criterion, res_f, res_t

    def show(self, features_names=None, class_name=None) -> Digraph:
        return self._abstract_show(True, features_names, class_name)
