from typing import NewType

import numpy as np

features = NewType("Array of feature", np.ndarray[float])
proba = NewType("Probability", float)
class_id = NewType("Id of Classes", int)
criterion = NewType("Criterion Type", float)


def shrunk_proba_vector(label_list: np.ndarray) -> np.ndarray:
    """ Return the list of probabilities of each occurring class in a random order"""
    return np.unique(label_list, return_counts=True)[1] / len(label_list)


def np_unique_to_proba_vector(np_unique_return, number_of_class: int) -> np.ndarray:
    """ Compute the vector of probabilities from the number of class present and the class
    distribution"""
    class_list, class_distribution = np_unique_return
    proba_vector = np.zeros(number_of_class)

    for i in range(len(class_list)):
        proba_vector[class_list[i]] = class_distribution[i]

    return proba_vector / np.sum(class_distribution)


# ----------------------------------- BAGGING -----------------------------------#

def subset_bagging(subset_size, f):
    """ Creates a feature subset of size subset_size """
    return np.random.choice(np.arange(f), subset_size)


# ----------------------------------- GINI CRITERION -----------------------------------#

def calculate_gini(y):
    """
    calculate gini index from array of class labels
    """
    return 1 - np.sum(shrunk_proba_vector(label_list=y) ** 2)


def calculate_mean_criterion(l_y, r_y, index_calculator):
    """ Calculate gini index from two different subset """
    l, r = index_calculator(l_y), index_calculator(r_y)
    return (l_y.size * l + r_y.size * r) / (l_y.size + r_y.size)


# ----------------------------------- LOG-LOSS CRITERION -----------------------------------#
def calculate_log_loss(y):
    """
    calculate log-loss index from array of class labels
    """
    proba_vect = shrunk_proba_vector(label_list=y)
    return np.sum(-proba_vect * np.log(proba_vect))
