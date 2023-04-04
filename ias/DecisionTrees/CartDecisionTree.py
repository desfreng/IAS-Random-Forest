import numpy as np
from graphviz import Digraph

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_mean_criterion, shrunk_proba_vector, subset_bagging


class DecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf, subset_size: int | None = None):
        super().__init__(max_depth, "gini")
        self._subset_size = subset_size

    def _find_threshold(self, data_set, label_set) -> tuple[float, int, int]:
        """
        Finds best threshold to split the dataset.
        The best (feature, threshold) is chosen between a random subset of y.
        """
        # TODO : Check this method with refactoring...
        x_copy = np.copy(data_set)
        best_gini = self.compute_criterion(data_set, label_set)

        if self._subset_size is None:
            bag = subset_bagging(int(np.sqrt(self.features_number)), self.features_number)
        else:
            bag = subset_bagging(self._subset_size, self.features_number)

        res_f, res_t = bag[0], x_copy[0][bag[0]]

        for f in bag:  # on itère sur les features d'un random subset
            l_x, r_x = np.empty(data_set.shape), x_copy[x_copy[:, f].argsort()]
            l_y, r_y = np.empty(label_set.shape), label_set[x_copy[:, f].argsort()]
            for i, e in enumerate(x_copy):
                np.add(l_x, e)
                r_x = r_x[i + 1:]
                new_gini = calculate_mean_criterion(l_y, r_y, self.compute_criterion)
                if new_gini < best_gini:
                    best_gini = new_gini
                    res_f, res_t = f, r_x[0][f]

        return best_gini, res_f, res_t

    def compute_criterion(self, data_set: np.ndarray, label_set: np.ndarray) -> float:
        """
        calculate gini index from array of class labels
        """
        return 1 - np.sum(shrunk_proba_vector(label_list=label_set) ** 2)

    def show(self, features_names=None, class_name=None) -> Digraph:
        return self._abstract_show(True, features_names, class_name)
