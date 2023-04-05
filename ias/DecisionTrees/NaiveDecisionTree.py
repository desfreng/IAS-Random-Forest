import numpy as np

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_mean_criterion


class NaiveDecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf, criterion_name: str = "gini"):
        super().__init__(max_depth, criterion_name)

    def _find_threshold(self, data_set, label_set) -> tuple[float, int, int]:
        """ Finds best threshold to split the dataset. """
        best_criterion = None
        best_feature = None
        best_threshold = None

        for feature in range(self.features_number):
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
