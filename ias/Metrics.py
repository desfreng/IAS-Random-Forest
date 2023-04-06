from typing import Optional

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np


def accuracy_score(known_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    nb_good_pred = np.count_nonzero(known_labels == predicted_labels)
    return nb_good_pred / len(predicted_labels)


def confusion_matrix(nb_class: int, known_labels: np.ndarray,
                     predicted_labels: np.ndarray) -> np.ndarray:
    conf_matrix = np.zeros((nb_class, nb_class), dtype=int)

    for i in range(nb_class):
        prediction_of_known_i = predicted_labels[np.argwhere(known_labels == i)]

        for j in range(nb_class):
            conf_matrix[i, j] = np.count_nonzero(prediction_of_known_i == j)

    return conf_matrix


def compute_luminance(color):
    def _compute(channel_value):
        if channel_value <= 0.04045:
            return channel_value / 12.92
        else:
            return ((channel_value + 0.055) / 1.055) ** 2.4

    return 0.2126 * _compute(color[0]) + 0.7152 * _compute(color[1]) + 0.0722 * _compute(color[2])


def _compute_best_contrast_color(background_color):
    color_luminance = compute_luminance(background_color)

    luminance_ratio_black_text = (color_luminance + 0.05) / 0.05
    luminance_ratio_white_text = 1.05 / (color_luminance + 0.05)

    if luminance_ratio_black_text > luminance_ratio_white_text:
        return "black"
    else:
        return "white"


def show_confusion_matrix(conf_mat: np.ndarray,
                          class_labels: Optional[list[str]] = None,
                          ax: Optional[matplotlib.axes.Axes] = None,
                          round_decimal: Optional[int] = None):
    if round_decimal is not None:
        conf_mat = conf_mat / np.max(conf_mat)
        max_value = 1.0
    else:
        max_value = np.max(conf_mat)

    text_kwargs = dict(ha='center', va='center')
    nb_class = len(conf_mat)

    if class_labels is None:
        class_labels = np.arange(nb_class)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    img = ax.imshow(conf_mat)
    cm = img.get_cmap()

    for i in range(nb_class):
        for j in range(nb_class):
            if round_decimal is not None:
                text_value = str(np.round(conf_mat[i, j], decimals=round_decimal))
            else:
                text_value = str(conf_mat[i, j])

            background_color = cm(conf_mat[i, j] / max_value)

            ax.text(j, i, text_value, **text_kwargs,
                    color=_compute_best_contrast_color(background_color))

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
