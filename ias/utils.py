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

def calculate_mean_criterion(l_y, r_y, index_calculator):
    """ Calculate gini index from two different subset """
    l, r = index_calculator(l_y), index_calculator(r_y)
    return (l_y.size * l + r_y.size * r) / (l_y.size + r_y.size)


#################################### LOG-LOSS CRITERION ####################################

def calculate_log_loss(y):
    """
    calculate log-loss index from array of class labels
    """
    log_loss, total, classes = 0, y.size, np.sort(y)
    cpt, current = 0, classes[0]
    for c in classes:
        if c != current:
            P = cpt / total
            log_loss += P*np.log(P)
            cpt = 1
            current = c
        else:
            cpt += 1
    return log_loss