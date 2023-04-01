import numpy as np
from graphviz import Digraph

from .utils import calculate_gini, calculate_mean_criterion, np_unique_to_proba_vector, \
    subset_bagging


class DecisionTree:
    def __init__(self, max_depth=np.inf, splitter="gini", subset_size=None):
        self._max_depth = max_depth
        self._splitter = splitter
        self._subset_size = subset_size

        self._fitted = False

        self._nodes = {}
        self._node_id = -1

        self._features_number = None
        self._class_number = None

    def _new_node_id(self):
        self._node_id += 1
        return self._node_id

    @property
    def nodes(self):
        return self._nodes

    def _find_threshold(self, x, y):
        """
        Finds best threshold to split the dataset.
        The best (feature, threshold) is chosen between a random subset of y.
        """
        x_copy = np.copy(x)
        if len(y) == 0:  # empty leaf
            return None, None, None, True
        best_gini = calculate_gini(y)
        if best_gini == 0:  # dataset pure
            return best_gini, None, None, True
        bag = subset_bagging(self._subset_size, x_copy[0].size)
        res_f, res_t = bag[0], x_copy[0][bag[0]]
        for f in bag:  # on itère sur les features d'un random subset
            l_x, r_x = np.empty(x.shape), x_copy[x_copy[:, f].argsort()]
            l_y, r_y = np.empty(y.shape), y[x_copy[:, f].argsort()]
            for i, e in enumerate(x_copy):
                np.add(l_x, e)
                r_x = r_x[i + 1:]
                new_gini = calculate_mean_criterion(l_y, r_y, calculate_gini)
                if new_gini < best_gini:
                    best_gini = new_gini
                    res_f, res_t = f, r_x[0][f]
        return best_gini, res_f, res_t, False

    def _check_for_fit(self):
        if not self._fitted:
            raise RuntimeError("RandomForest must be fitted")

    def fit_bis(self, x, y, current_node_id, remaining_depth) -> None:
        if remaining_depth == 0:  # condition d'arrêt : max_depth atteinte
            return
        else:
            # todo : just adapt with the name of the splitter (only 2 cases : random or other)
            if self._splitter == "gini":
                g, f, t, b = self._find_threshold(x, y)
                if (remaining_depth == 1) or b:  # le nœud est une feuille
                    class_list, class_distribution = np.unique(y, return_counts=True)
                    self._nodes[current_node_id] = {
                        "is_node": False,
                        "criterion": g,
                        "class_list": class_list,
                        "class_distribution": class_distribution,
                    }
                else:
                    x_a = x[np.argwhere(x[:, f] <= t).flatten(), :]
                    x_b = x[np.argwhere(x[:, f] > t).flatten(), :]
                    y_a = y[np.argwhere(x[:, f] <= t).flatten()]
                    y_b = y[np.argwhere(x[:, f] > t).flatten()]
                    id1 = self._new_node_id()
                    id2 = self._new_node_id()
                    self._nodes[current_node_id] = {"is_node": True,
                                                    "criterion": g,
                                                    "feature": f,
                                                    "threshold": t,
                                                    "left_son_id": id1,
                                                    "right_son_id": id2}
                    self.fit_bis(x_a, y_a, id1, remaining_depth - 1)
                    self.fit_bis(x_b, y_b, id2, remaining_depth - 1)

            elif self._splitter == "random":
                raise ValueError("random not implemented yet, stay tuned.")
            elif self._splitter == "entropy":
                raise ValueError("entropy not implemented yet, stay tuned.")
            elif self._splitter == "log_loss":
                raise ValueError("log_loss not implemented yet, stay tuned.")
            else:
                raise ValueError("splitter parameter must be gini or random")

    def fit(self, x, y) -> None:
        """ Crée un arbre avec les données labellisées (x, y) """
        self._features_number = len(x[0])
        self._class_number = np.max(y)

        if self._subset_size is None:
            self._subset_size = int(np.sqrt(x[0].size))

        self.fit_bis(x, y, self._new_node_id(), self._max_depth)
        self._fitted = True

    def predict_proba(self, x) -> "proba of class x for each elm":
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        pass

    # def predict_bis(self, x, node):
    #    if not self._nodes[node]["is_node"]: # condition d'arrêt : feuille
    #        return 

    def predict(self, x) -> "y like":
        """ Prend des données non labellisées puis renvoi les labels estimés """
        self._check_for_fit()
        return np.argmax(self.predict_proba(x), axis=1).reshape(-1, 1)

    def show(self, features_names=None, class_name=None) -> Digraph:
        """ Affiche le Tree (Utilisable dans Jupyter Notebook)"""
        if self._fitted:
            return Digraph()

        if features_names is None:
            features_names = list(map(str, range(self._features_number)))

        if class_name is None:
            features_names = list(map(str, range(self._class_number)))

        splitting_node_args = {"shape": "ellipse"}
        leaf_args = {"shape": "rectangle", "style": "rounded"}

        dot_tree = Digraph()
        for node_id, node_data in self._nodes.items():
            criterion_value = node_data["criterion"]
            if node_data["is_node"]:
                feature_name = features_names[node_data["feature"]]
                threshold = node_data["threshold"]

                node_str = f"{feature_name} ≤ {threshold}\n" \
                           f"{self._splitter} = {criterion_value}"
                node_args = splitting_node_args

            else:
                class_distribution = node_data["class_distribution"]
                majority_class = class_name[np.argmax(class_distribution)]
                proba_vector = np_unique_to_proba_vector(node_data["class_list"],
                                                         class_distribution,
                                                         self._class_number)

                node_str = f"{self._splitter} = {criterion_value}\n" \
                           f"sample = {np.sum(class_distribution)}\n" \
                           f"Probabilities : {proba_vector}\n" \
                           f"Majority Class = {majority_class}"
                node_args = leaf_args

            dot_tree.node(str(node_id), node_str, **node_args)

        for node_id, node_data in self._nodes.items():
            if node_data["is_node"]:
                left_son = node_data["left_son_id"]
                right_son = node_data["right_son_id"]
                dot_tree.edges([(str(node_id), str(left_son)), (str(node_id), str(right_son))])

        return dot_tree
