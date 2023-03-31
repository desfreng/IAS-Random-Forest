import numpy as np


def accuracy_score(known_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    nb_good_pred = np.count_nonzero(known_labels == predicted_labels)
    return nb_good_pred / len(predicted_labels)


def confusion_matrix(nb_class: int, known_labels: np.ndarray,
                     predicted_labels: np.ndarray) -> np.ndarray:
    conf_matrix = np.zeros((nb_class, nb_class))

    for i in range(nb_class):
        prediction_of_known_i = predicted_labels[np.argwhere(known_labels == i)]

        for j in range(nb_class):
            conf_matrix[i, j] = np.count_nonzero(prediction_of_known_i == j)

    return conf_matrix


def show_confusion_matrix(conf_mat: np.ndarray) -> None:
    # Todo : Complete this Function
    pass
