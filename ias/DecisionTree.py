import numpy as np
import math

class DecisionTree:
    def __init__(self):
        self._nodes = {}
        self._leaves = {}
        self._node_id = 0

    def _new_node_id(self):
        self._node_id += 1
        return self._node_id
    
    def calculate_gini(y):
        """
        calculate gini index from array of class labels
        """
        gini, total, C = 1, y.size, np.sort(y)
        cpt, current = 0, C[0]
        for x in C:
            if x != current:
                gini -= (cpt / total)**2
                cpt = 1
                current = x
            else:
                cp += 1
        return gini

    def calculate_mean_gini(l_y, r_y):
        l_gini, r_gini = calculate_gini(l_y), calculate_gini(r_y)
        return (l_y.size * l_gini + r_y.size * r_gini) / (l_y.size + r_y.size)

    def subset_bagging(subset_size, f):
        return np.random.choice(np.arange(f), subset_size)

    def find_treshold(x, y, subset_size):
        """
        Finds best treshold to split the dataset.
        The best (feature, treshold) is chosen between a random subset of y.
        """
        x_copy = np.copy(x)
        bag = _subset_bagging(subset_size,x_copy[0].size)
        best_gini = calculate_gini(y)
        if best_gini == 0: # dataset pure
            return None, None, True
        res_f, res_t = bag[0], x_copy[0][bag[0]]
        for f in bag: # on itere sur les features d'un random subset
            l, r = np.empty(0), np.sort(x_copy)
            for i, e in enumerate(x_copy):
                np.add(l, np.array(e))
                r = r[i+1:]
                new_gini = calculate_mean_gini(l_y, r_y)
                if new_gini < best_gini:
                    best_gini = new_gini
                    res_f, res_t = f, r[0][f]
        return res_f, res_t, False

    def fit_bis(self, x, y, max_depth, splitter, subset_size, id)-> None:
        if max_depth == 0: # condition d'arret : max_depth atteinte
            return
        else:
            if splitter == "gini":
                f, t, b = _find_treshold(x, y, subset_size)
                if (max_depth == 1) or b: # le noeud est une feuille
                    self._nodes[id] = [False, x, y]
                else:
                    condition = x[f] < t
                    x_a = x[f][condition]
                    x_b = x[f][not condition]
                    y_a = np.argwhere(condition)
                    y_b = np.argwhere(not condition)
                    id1 = self._new_node_id()
                    id2 = self._new_node_id()
                    self._nodes[id] = [True, f, t, id1, id2]
                    self.fit_bis(x_a, y_a, max_depth-1, splitter, subset_size)
                    self.fit_bis(x_b, y_b, max_depth-1, splitter, subset_size)

            elif splitter == "random":
                raise ValueError("random not implemented yet, stay tuned.")
            else:
                raise ValueError("splitter parameter must be gini or random")

    def fit(self, x, y, max_depth=None, splitter="gini", subset_size=None) -> None:
        """ Crée un arbre avec les données labellisées (x, y) """
        if subset_size == None:
                subset_size = np.sqrt(x[0].size)
        self.fit_bis(x, y, max_depth, splitter, subset_size, self._new_node_id())

    def predict(self, x)-> "y like":
        """ Prend des données non labellisées puis renvoi les labels estimés """
        pass

    def show(self) -> None:
        """ Affiche le Tree (graphviz ?) """
        pass

    def predict_proba(self, x) -> "proba of class x for each elm":
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        pass

    def subset_size(set_size) -> int:
        return math.sqrt(set_size)