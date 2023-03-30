import numpy as np
import math

class DecisionTree:
    def __init__(self):
        self._nodes = {}
        self._leaves = {}
    
    def calculate_gini(y):
        """
        calculate gini index from array of class labels
        """
        gini, total, C = 1, y.size, np.sort(Y)
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
        return res_f, res_t

    def fit(self, x, y, max_depth=None, splitter="gini", subset_size=None) -> None:
        """ Crée un arbre avec les données labellisées (x, y) """
        if max_depth == 0: # condition d'arret : max_depth atteinte
            return
        else:
            if subset_size == None:
                subset_size = np.sqrt(x[0].size)
            if splitter == "gini":
                f, t = _find_treshold(x, y, subset_size)
                son_a = np.extract(x[f] < t, x[f])
                son_b = np.extract(x[f]>= t, x[f])

            elif splitter == "random":
                raise ValueError("random not implemented yet, stay tuned.")
            else:
                raise ValueError("splitter parameter must be gini or random")



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