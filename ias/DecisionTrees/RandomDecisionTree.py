from typing import Optional

import numpy as np

from ..AbstractDecisionTree import AbstractDecisionTree
from ..utils import calculate_mean_criterion, criterion, random, subset_bagging


class RandomDecisionTree(AbstractDecisionTree):
    def __init__(self, max_depth=np.inf, subset_size: Optional[int] = None,
                 criterion_name: str = "gini"):
        super().__init__(max_depth, criterion_name)
        self._subset_size = subset_size

    @staticmethod
    def _threshold_array(attr_bag):
        return random(np.min(attr_bag), np.max(attr_bag))

    def _find_threshold(self, data_set, label_set) -> tuple[criterion, int, float]:
        """
        Chooses best threshold to split the dataset between random threshold for each feature.
        Random subspaces is applied to y.
        :param data_set: the dataset
        :param label_set: the label_set
        :return: tuple containing (the best criterion value, feature number, threshold value)
        """

        if self._subset_size is None:
            bag = subset_bagging(int(np.sqrt(self.features_number)), self.features_number)
        else:
            bag = subset_bagging(self._subset_size, self.features_number)

        thresholds = np.apply_along_axis(self._threshold_array, 1,
                                         np.transpose(data_set[:, bag]))

        best_criterion = None
        best_feature = None
        best_threshold = None

        for f_id, threshold in enumerate(thresholds):
            feature = bag[f_id]

            feature_data = data_set[:, feature].flatten()
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
