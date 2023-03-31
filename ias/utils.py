import numpy as np


#################################### BAGGING ####################################

def subset_bagging(subset_size, f):
    """ Creates a feature subset of size subset_size """
    return np.random.choice(np.arange(f), subset_size)

#################################### GINI CRITERION ####################################

def calculate_gini(y):
    """
    calculate gini index from array of class labels
    """
    gini, total, classes = 1, y.size, np.sort(y)
    cpt, current = 0, classes[0]
    for c in classes:
        if c != current:
            gini -= (cpt / total)**2
            cpt = 1
            current = c
        else:
            cpt += 1
    return gini

def calculate_mean_gini(l_y, r_y):
    """ Calculate gini index from two different subset """
    l_gini, r_gini = calculate_gini(l_y), calculate_gini(r_y)
    return (l_y.size * l_gini + r_y.size * r_gini) / (l_y.size + r_y.size)

