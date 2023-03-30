import numpy as np
import math

def calculate_gini(Y):
    """
    calculate gini index from array of class labels
    """
    gini, total, C = 1, Y.size, np.sort(Y)
    cpt, current = 0, C[0]
    for x in C:
        if x != current:
            gini -= (cpt / total)**2
            cpt = 1
            current = x
        else:
            cp += 1
    return gini

    def _calculate_mean_gini(l_Y, r_Y):
        l_gini, r_gini = calculate_gini(l_Y), calculate_gini(r_Y)
        return (l_Y.size * l_gini + r_Y.size * r_gini) / (l_Y.size + r_Y.size)

    def _subset_bagging(subset_size, F):
        return np.random.choice(np.arange(F), subset_size)

    def _find_treshold(X, Y, subset_size):
        """
        Finds best treshold to split the dataset.
        The best (feature, treshold) is chosen between a random subset of Y.
        """
        X_copy = np.copy(X)
        B = _subset_bagging(subset_size,X_copy[0].size)
        best_gini = calculate_gini(Y)
        res_f, res_t = B[0], X_copy[0][B[0]]
        for f in B: # on itere sur les features d'un random subset
            l, r = np.empty(0), np.sort(X_copy)
            for i, x in enumerate(X_copy):
                np.add(l, np.array(x))
                r = r[i+1:]
                new_gini = calculate_mean_gini(l_Y, r_Y)
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