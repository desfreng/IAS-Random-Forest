import matplotlib.pyplot as plt
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


def show_confusion_matrix(conf_mat: np.ndarray, class_labels=None, ax=None):
    text_kwargs = dict(ha='center', va='center', color='black')
    nb_class = len(conf_mat)

    if class_labels is None:
        class_labels = np.arange(nb_class)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    img = ax.imshow(conf_mat)

    for i in range(nb_class):
        for j in range(nb_class):
            ax.text(j, i, str(conf_mat[i, j]), **text_kwargs)

    fig.colorbar(img, ax=ax)

    ax.set(
        xticks=np.arange(nb_class),
        yticks=np.arange(nb_class),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel="Known labels",
        xlabel="Predicted labels",
    )

    return fig, ax
