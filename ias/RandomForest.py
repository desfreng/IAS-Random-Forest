from typing import Type

import numpy as np
from numpy import ndarray

from .AbstractDecisionTree import AbstractDecisionTree
from .utils import attributes, class_id, proba


class RandomForest:
    def __init__(self, tree_class: Type[AbstractDecisionTree], tree_number: int,
                 subset_size: int, **args):
        self._subset_size = subset_size
        self._trees = []

        for _ in range(tree_number):
            self._trees.append(tree_class(**args))

        self._fitted = False

    def _check_for_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("RandomForest must be fitted")

    def fit(self, data_set: np.ndarray[attributes], label_set: np.ndarray[class_id]) -> None:
        """ Crée des arbres avec les données labellisées (x, y) """
        indices = np.arange(len(data_set))

        for tree in self._trees:
            subset_indices = np.random.choice(indices, size=self._subset_size, replace=True)
            tree.fit(data_set[subset_indices], label_set[subset_indices])

        self._fitted = True

    def predict(self, data_to_classify: np.ndarray[attributes]) -> ndarray[int]:
        """ Prend des données non labellisées puis renvoi les labels estimés """
        self._check_for_fit()
        return np.argmax(self.predict_proba(data_to_classify), axis=1)

    def predict_proba(self, data_to_classify: np.ndarray[attributes]) -> np.ndarray[proba]:
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        self._check_for_fit()

        proba_sum = None
        for tree in self._trees:
            if proba_sum is None:
                proba_sum = tree.predict_proba(data_to_classify)
            else:
                proba_sum += tree.predict_proba(data_to_classify)
                
        proba_sum /= len(self._trees)
        return proba_sum
