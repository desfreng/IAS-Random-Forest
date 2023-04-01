import numpy as np
from utils import calculate_gini, calculate_mean_criterion, subset_bagging


class DecisionTree:
    def __init__(self):
        self._nodes = {}
        self._leaves = {}
        self._node_id = -1

    def _new_node_id(self):
        self._node_id += 1
        return self._node_id
    
    @property
    def nodes(self):
        return self._nodes

    def _find_threshold(self, x, y, subset_size):
        """
        Finds best threshold to split the dataset.
        The best (feature, threshold) is chosen between a random subset of y.
        """
        x_copy = np.copy(x)
        if len(y) == 0: # empty leaf
            return None, None, None, True
        best_gini = calculate_gini(y)
        if best_gini == 0:  # dataset pure
            return best_gini, None, None, True
        bag = subset_bagging(subset_size, x_copy[0].size)
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

    def fit_bis(self, x, y, max_depth, splitter, subset_size, id) -> None:
        if max_depth == 0:  # condition d'arrêt : max_depth atteinte
            return
        else:
            # todo : just adapt with the name of the splitter (only 2 cases : random or other)
            if splitter == "gini":
                g, f, t, b = self._find_threshold(x, y, subset_size)
                if (max_depth == 1) or b:  # le nœud est une feuille
                    _, d = np.unique(y, return_counts=True)
                    self._nodes[id] = {"is_node": False, "gini": g, "samples": len(y), "distribution": d}
                else:
                    x_a = x[np.argwhere(x[:, f]<= t).flatten(), :]
                    x_b = x[np.argwhere(x[:, f] > t).flatten(), :]
                    y_a = y[np.argwhere(x[:, f]<= t).flatten()]
                    y_b = y[np.argwhere(x[:, f] > t).flatten()]
                    id1 = self._new_node_id()
                    id2 = self._new_node_id()
                    self._nodes[id] = {"is_node": True, "gini": g, "feature": f, "treshold": t, "son_1_id": id1, "son_2_id": id2}
                    self.fit_bis(x_a, y_a, max_depth - 1, splitter, subset_size, id1)
                    self.fit_bis(x_b, y_b, max_depth - 1, splitter, subset_size, id2)

            elif splitter == "random":
                raise ValueError("random not implemented yet, stay tuned.")
            elif splitter == "entropy":
                raise ValueError("entropy not implemented yet, stay tuned.")
            elif splitter == "log_loss":
                raise ValueError("log_loss not implemented yet, stay tuned.")
            else:
                raise ValueError("splitter parameter must be gini or random")

    def fit(self, x, y, max_depth=np.inf, splitter="gini", subset_size=None) -> None:
        """ Crée un arbre avec les données labellisées (x, y) """
        if subset_size == None:
            subset_size = int(np.sqrt(x[0].size))
        self.fit_bis(x, y, max_depth, splitter, subset_size, self._new_node_id())

    def predict(self, x) -> "y like":
        """ Prend des données non labellisées puis renvoi les labels estimés """
        pass

    def show(self) -> None:
        """ Affiche le Tree (graphviz ?) """
        pass

    def predict_proba(self, x) -> "proba of class x for each elm":
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        pass