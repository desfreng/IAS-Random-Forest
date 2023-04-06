from typing import NewType

import numpy as np

attributes = NewType("Array of attributes", np.ndarray[float])
proba = NewType("Probability", float)
class_id = NewType("Id of Classes", int)


def shrunk_proba_vector(label_list: np.ndarray) -> np.ndarray:
    """ Return the list of probabilities of each occurring class in a random order"""
    return np.unique(label_list, return_counts=True)[1] / len(label_list)


def np_unique_to_proba_vector(np_unique_return, number_of_class: int) -> np.ndarray:
    """ Compute the vector of probabilities from the number of class present and the class
    distribution"""
    class_list, class_distribution = np_unique_return
    proba_vector = np.zeros(number_of_class)

    for i, cls_id in enumerate(class_list):
        proba_vector[int(cls_id)] = class_distribution[i]
        
    return proba_vector / np.sum(class_distribution)


def random(minimum, maximum):
    return (np.random.rand() * 0.98 + 0.01) * (maximum - minimum) + minimum


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

# ----------------------------------- PRUNING -----------------------------------#

def compute_tree_score(nodes):
    """ Compute efficiency score from tree """
    return 1

def create_leaf_from_class(cls, samples, nb_cls):
    """ Create leaf dictionnary from """
    proba_vector = np.zeros(nb_cls)
    proba_vector[cls] = 1
    leaf = {
        "is_node": False,
        "probability_vector": proba_vector,
        "samples": samples,
        "criterion": 1
    }
    return leaf

def acceptable(new_score, old_score):
    """Returns whether new_score is acceptable compared to old_score or not"""
    return new_score + 1e-4 >= old_score