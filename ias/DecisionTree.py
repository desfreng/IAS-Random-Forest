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

def find_treshold(X, Y):
    """
    find best treshold to split the dataset
    """
    best_gini = calculate_gini(Y)
    for f in range(X[0].size): # on itere sur les features
        for x in X:
            l, r = [], X.copy()
            gini = calculate_gini()
            if gini < best_gini:
                best_gini = gini
                # res_f, res_t =

    return res_f, res_t

import numpy as np
from typing import type

# Label = typing.


class DecisionTree:
    def __init__(self):
        pass

    def fit(self, x, y) -> None:
        """ Crée un arbre avec les données labellisées (x, y) """
        pass

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