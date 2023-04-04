from abc import ABC, abstractmethod

import numpy as np
from graphviz import Digraph
from numpy import ndarray

from .utils import class_id, criterion, attributes, np_unique_to_proba_vector, proba


class AbstractDecisionTree(ABC):
    def __init__(self, max_depth: int = np.inf, criterion_name: None | str = None):
        self._max_depth = max_depth
        self._criterion_name = criterion_name

        self._fitted = False

        self._nodes = {}
        self._node_id = -1

        self._features_number: None | int = None
        self._class_number: None | int = None

    @property
    def features_number(self) -> int:
        return self._features_number

    @property
    def class_number(self) -> int:
        return self._class_number

    def _check_for_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("RandomForest must be fitted")

    def _new_node_id(self) -> int:
        self._node_id += 1
        return self._node_id

    @abstractmethod
    def compute_criterion(self, label_set: np.ndarray[class_id]) -> criterion:
        pass

    @abstractmethod
    def _find_threshold(self, data_set: np.ndarray[attributes], label_set: np.ndarray[class_id]) \
            -> tuple[criterion, int, float]:
        """ Find the best feature and threshold to cut `data_set` in two non-empty parts.

        :param data_set: Array of Features
        :param label_set:
        :return: a tuple with criterion, the feature's id and the threshold value
        """
        pass

    @staticmethod
    def _is_label_set_pure(label_set: np.ndarray[class_id]) -> bool:
        return len(np.unique(label_set)) == 1

    def _fit_bis(self, data_set: np.ndarray[attributes], label_set: np.ndarray[class_id],
                 current_node_id: int, remaining_depth: int) -> None:
        if (remaining_depth == 1) or (self._is_label_set_pure(label_set)):
            # le nœud est une feuille
            probability_vector = np_unique_to_proba_vector(np.unique(label_set, return_counts=True),
                                                           self._class_number)

            self._nodes[current_node_id] = {
                "is_node": False,
                "probability_vector": probability_vector,
                "samples": len(label_set),
                "criterion": self.compute_criterion(label_set)
            }
        else:
            criterion_value, selected_feature, threshold = self._find_threshold(data_set, label_set)
            left_indexes = np.argwhere(data_set[:, selected_feature] <= threshold).flatten()
            right_indexes = np.argwhere(data_set[:, selected_feature] > threshold).flatten()

            left_son_id = self._new_node_id()
            right_son_id = self._new_node_id()

            self._nodes[current_node_id] = {"is_node": True,
                                            "criterion": criterion_value,
                                            "feature": selected_feature,
                                            "threshold": threshold,
                                            "left_son_id": left_son_id,
                                            "right_son_id": right_son_id,
                                            "samples": len(label_set)}

            self._fit_bis(data_set[left_indexes, :], label_set[left_indexes], left_son_id,
                          remaining_depth - 1)
            self._fit_bis(data_set[right_indexes, :], label_set[right_indexes], right_son_id,
                          remaining_depth - 1)

    def fit(self, data_set: np.ndarray[attributes], label_set: np.ndarray[class_id]) -> None:
        """ Crée un arbre avec les données labellisées (x, y) """
        self._features_number = len(data_set[0])
        self._class_number = int(np.max(label_set)) + 1

        assert self._max_depth > 0

        self._fit_bis(data_set, label_set, self._new_node_id(), self._max_depth)
        self._fitted = True

    def _classify(self, elm_to_classify: attributes, node_id: int) -> np.ndarray[proba]:
        """
        Recursively classify a single element passing through nodes.
        Returns probability array
        """
        if not self._nodes[node_id]["is_node"]:  # condition d'arrêt : feuille
            return self._nodes[node_id]["probability_vector"]
        else:
            f, t = self._nodes[node_id]["feature"], self._nodes[node_id]["threshold"]
            if elm_to_classify[f] <= t:
                return self._classify(elm_to_classify, self._nodes[node_id]["left_son_id"])
            else:
                return self._classify(elm_to_classify, self._nodes[node_id]["right_son_id"])

    def predict_proba(self, data_to_classify: np.ndarray[attributes]) -> np.ndarray[proba]:
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        self._check_for_fit()
        return np.apply_along_axis(self._classify, 1, data_to_classify, 0)

    def predict(self, data_to_classify: np.ndarray[attributes]) -> ndarray[int]:
        """ Prend des données non labellisées puis renvoi les labels estimés """
        self._check_for_fit()
        return np.argmax(self.predict_proba(data_to_classify), axis=1)

    @abstractmethod
    def show(self, features_names: list[str] = None, class_name: list[str] = None) -> Digraph:
        pass

    def _abstract_show(self, show_criterion: bool, features_names: list[str] = None,
                       class_name: list[str] = None) -> Digraph:
        """ Affiche le Tree (Utilisable dans Jupyter Notebook)"""
        if not self._fitted:
            return Digraph()

        if features_names is None:
            features_names = list(map(lambda i: f"Feature {i + 1}", range(self._features_number)))

        if class_name is None:
            class_name = list(map(lambda i: f"Class {i}", range(self._class_number)))

        splitting_node_args = {"shape": "ellipse"}
        leaf_args = {"shape": "rectangle", "style": "rounded"}

        dot_tree = Digraph()
        for node_id, node_data in self._nodes.items():
            criterion_value = np.round(node_data["criterion"], 2)
            samples = node_data["samples"]

            if node_data["is_node"]:
                feature_name = features_names[node_data["feature"]]
                threshold = np.round(node_data["threshold"], 2)

                node_str = f"{feature_name} ≤ {threshold}\n"
                if show_criterion:
                    node_str += f"{self._criterion_name} = {criterion_value}\n"

                node_str += f"Samples = {samples}"
                node_args = splitting_node_args

            else:
                proba_vector = node_data["probability_vector"]
                majority_class = class_name[np.argmax(proba_vector)]

                if show_criterion:
                    node_str = f"{self._criterion_name} = {criterion_value}\n"
                else:
                    node_str = ""

                node_str += f"Samples = {samples}\n" \
                            f"Probabilities : {np.round(proba_vector, 2)}\n" \
                            f"Majority Class = {majority_class}"
                node_args = leaf_args

            dot_tree.node(str(node_id), node_str, **node_args)

        for node_id, node_data in self._nodes.items():
            if node_data["is_node"]:
                left_son = node_data["left_son_id"]
                right_son = node_data["right_son_id"]
                dot_tree.edges([(str(node_id), str(left_son)), (str(node_id), str(right_son))])

        return dot_tree
