import numpy as np

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
    best_gini = calculate_gini
    for f in range(X[0].size):
        for x in X:
            l, r = 
            gini = calculate_gini
            if gini < best_gini:
                best_gini = gini
                res_f, res_t =
        
    return res_f, res_t
