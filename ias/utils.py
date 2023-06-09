from typing import NewType, Union

import numpy as np

from ias.Datasets import Dataset

attributes = NewType("Array of attributes", np.ndarray[float])
proba = NewType("Probability", float)
class_id = NewType("Id of Classes", int)


def split_dataset(ratio: Union[int, float], dataset: Dataset) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(dataset.attributes))
    np.random.shuffle(indices)
    if isinstance(ratio, float):
        train_size = int(len(dataset.attributes) * ratio)
    elif isinstance(ratio, int):
        train_size = ratio
    else:
        raise ValueError("Error")

    return (dataset.attributes[indices[:train_size]], dataset.labels[indices[:train_size]],
            dataset.attributes[indices[train_size:]], dataset.labels[indices[train_size:]])


# ----------------------------------- PROBABILITY VECTORS -----------------------------------#

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


def is_label_set_pure(label_set: np.ndarray[class_id]) -> bool:
    return len(np.unique(label_set)) == 1


# ----------------------------------- CRITERION COMPUTING -----------------------------------#

def calculate_mean_criterion(l_y, r_y, index_calculator):
    """ Calculate gini index from two different subset """
    l, r = index_calculator(l_y), index_calculator(r_y)
    return (l_y.size * l + r_y.size * r) / (l_y.size + r_y.size)


# ----------------------------------- GINI CRITERION -----------------------------------#
def calculate_gini(y):
    """
    calculate gini index from array of class labels
    """
    return 1 - np.sum(shrunk_proba_vector(label_list=y) ** 2)


# ----------------------------------- LOG-LOSS CRITERION -----------------------------------#
def calculate_log_loss(y):
    """
    calculate log-loss index from array of class labels
    """
    proba_vect = shrunk_proba_vector(label_list=y)
    return np.sum(-proba_vect * np.log(proba_vect))
