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
        best_criterion = 2

        if self._subset_size is None:
            bag = subset_bagging(int(np.sqrt(self.features_number)), self.features_number)
        else:
            bag = subset_bagging(self._subset_size, self.features_number)

        res_f, res_t = bag[0], data_set[0][bag[0]]
        r_x = np.empty(0)
        for f in bag:  # on itère sur les features d'un random subset
            r_x = data_set[data_set[:, f].argsort()]
            l_x = np.empty(data_set.shape)
            l_y, r_y = np.empty(label_set.shape), label_set[data_set[:, f].argsort()]
            for i, e in enumerate(data_set):
                np.add(l_x, e)
                r_x = r_x[i + 1:]
                new_crit = calculate_mean_criterion(l_y, r_y, self.compute_criterion)
                if new_crit < best_criterion:
                    best_criterion = new_crit
                    res_f, res_t = f, r_x[0][f]
        if r_x.size == 0:
            print(data_set)
            raise FileExistsError
            # return self._find_threshold(data_set, label_set)

        return best_criterion, res_f, res_t

    def show(self, features_names=None, class_name=None) -> Digraph:
        return self._abstract_show(True, features_names, class_name)
