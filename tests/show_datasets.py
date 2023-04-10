import random
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).parent.parent))

from ias import PCA, RandomForest, split_dataset
from ias.Datasets import Emnist, Iris
from ias.Metrics import accuracy_score, confusion_matrix, cross_validation, show_confusion_matrix

from scipy import io as sio


def show_iris():
    pca = PCA(2)
    pca.fit(Iris.attributes)
    x = pca.compress(Iris.attributes)
    y = Iris.labels
    cmap = plt.get_cmap()

    for class_id, class_name in enumerate(Iris.class_names):
        to_take = y == class_id
        plt.scatter(x[to_take, 0], x[to_take, 1], c=cmap(y[to_take] / 2), marker="x",
                    label=class_name)

    plt.legend()
    plt.title("Iris Dataset (PCA over 2 components)")
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.savefig("IrisPCA.png", format="png", dpi=384)
    plt.clf()


def show_emnist(emnist_letter_mat: str):
    mat = sio.loadmat(emnist_letter_mat)
    data = mat['dataset']
    x_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]

    def get_mapping(char):
        return np.argwhere(ord(char) == data["mapping"][0, 0])[0, 0] + 1

    def gen_img():
        img = np.zeros((28 * 2, 28 * 6))

        for img_id, img_label in enumerate("enmist"):
            letter_pos = random.choice(
                np.argwhere(y_train.flatten() == get_mapping(img_label)).flatten())
            letter_data = x_train[letter_pos].reshape(28, 28).T
            img[:28, 28 * img_id:28 * (img_id + 1)] = letter_data

        for img_id, img_label in enumerate("digits"):
            letter_pos = random.choice(
                np.argwhere(y_train.flatten() == get_mapping(img_label)).flatten())
            letter_data = x_train[letter_pos].reshape(28, 28).T
            img[28:, 28 * img_id:28 * (img_id + 1)] = letter_data

        return img

    for i in range(10):
        plt.imshow(gen_img())
        plt.savefig(f"ENMIST_{i}.png", format="png", dpi=384)
        plt.clf()


def show_distribution():
    probabilities = [np.array([0.33, 0.33, 0.34]),
                     np.array([0.6, 0.3, 0.1]),
                     np.array([0.8, 0.2, 0.0]),
                     np.array([1., 0.0, 0.0])]
    for i, proba_vect in enumerate(probabilities):
        assert abs(sum(proba_vect) - 1) < 1e-6

        n_class = len(proba_vect)
        gini_index = 1 - (proba_vect ** 2).sum()

        plt.figure(figsize=(5, 6))

        plt.bar(np.arange(1, n_class + 1), proba_vect, width=0.4)

        plt.title(f"Class Distribution\n$gini$ impurity : ${gini_index:.2f}$")
        plt.xticks(np.arange(1, 1 + n_class))
        plt.ylim(0, 1)
        plt.xlabel("Possible Class")
        plt.ylabel("Probability of Class")
        plt.savefig(f"distribution_{i}.png", format="png", dpi=384)
        plt.clf()


def rf_worker(task_queue, done_queue, emnist_compressed_train_x, emnist_train_y, nb_test):
    for tree_number, max_depth, training_subset_size in iter(task_queue.get, 'STOP'):
        print(f"Forest with : tree_number={tree_number}, max_depth={max_depth}, "
              f"training_subset_size={training_subset_size}...")
        forest = RandomForest(tree_number=tree_number, max_depth=max_depth,
                              training_subset_size=training_subset_size, do_bagging=True,
                              class_number=10)
        score_vect = cross_validation(forest, emnist_compressed_train_x, emnist_train_y, nb_test)
        done_queue.put([tree_number, max_depth, training_subset_size, score_vect])


def optimize_rf():
    number_of_processes = 20
    emnist_train_x, emnist_train_y, _, _ = split_dataset(1000, Emnist)
    pca = PCA(0.9)

    pca.fit(emnist_train_x)
    emnist_compressed_train_x = pca.compress(emnist_train_x)

    tree_numbers = 2 ** np.arange(2, 9)
    max_depths = 2 ** np.arange(5)
    training_subset_sizes = 2 ** np.arange(3, 10)
    nb_test = 5

    scores = np.empty((tree_numbers.size, max_depths.size, training_subset_sizes.size, nb_test))
    nb_tasks = tree_numbers.size * max_depths.size * training_subset_sizes.size

    task_queue = Queue()

    for tree_number in tree_numbers:
        for max_depth in max_depths:
            for training_subset_size in training_subset_sizes:
                task_queue.put((tree_number, max_depth, training_subset_size))

    done_queue = Queue()
    for _ in range(number_of_processes):
        Process(target=rf_worker, args=(task_queue, done_queue, emnist_compressed_train_x,
                                        emnist_train_y, nb_test)).start()

    for _ in range(nb_tasks):
        tree_number, max_depth, training_subset_size, score_vect = done_queue.get()
        tree_index = np.where(tree_numbers == tree_number)[0]
        depth_index = np.where(max_depths == max_depth)[0]
        subset_index = np.where(training_subset_sizes == training_subset_size)[0]

        scores[tree_index, depth_index, subset_index, :] = score_vect

    np.save(str(Path(__file__).parent / "random_forest_data.npy"), scores)

    for _ in range(number_of_processes):
        task_queue.put('STOP')


def max_depth_accuracy_section(selected_depth: int,
                               x_annotation: Optional[int] = None,
                               y_annotation: Optional[int] = None):
    tree_numbers = 2 ** np.arange(2, 9)
    training_subset_sizes = 2 ** np.arange(3, 10)
    max_depths = 2 ** np.arange(5)

    scores = np.load(str(Path(__file__).parent / "random_forest_data.npy"))
    data = scores[:, selected_depth, :].swapaxes(0, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    filename = f"accuracy_slice_depth_{selected_depth}.png"

    co = ax.contourf(training_subset_sizes, tree_numbers, data.mean(axis=2))
    if x_annotation is not None and y_annotation is not None:
        ax.plot([training_subset_sizes[x_annotation]], [tree_numbers[y_annotation]], 'o',
                color="red")
        ax.annotate(
            f"{data[x_annotation, y_annotation].mean():.2f} "
            f"(std {data[x_annotation, y_annotation].std():.2f})",
            xy=(training_subset_sizes[x_annotation], tree_numbers[y_annotation]), xycoords='data',
            ha="center", va="top", xytext=(0, 30), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="->"))
        filename = f"accuracy_slice_depth_{selected_depth}_annotated.png"

    ax.set_ylabel(r"Number of trees per forest $(log\ scale)$")
    ax.set_yscale('log')
    ax.set_yticks(tree_numbers)
    ax.set_ylim([tree_numbers.max(), tree_numbers.min()])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel(r"Size of training subsets $(log\ scale)$")
    ax.set_xscale('log')
    ax.set_xticks(training_subset_sizes)
    ax.set_xlim([training_subset_sizes.min(), training_subset_sizes.max()])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_title("Accuracy score by the number of trees per forest\n"
                 "and the size of the training subsets\n"
                 f"Maximum tree depth : {max_depths[selected_depth]}")

    fig.colorbar(mappable=co, ax=ax, pad=0.1)

    fig.savefig(filename, format="png", dpi=384)
    plt.show()


def rf_cub_accuracy(selected_depth: Optional[int] = None):
    x = np.arange(2, 9)  # tree_numbers
    y = np.arange(3, 10)  # training_subset_sizes
    z = np.arange(5)  # max_depths

    data = np.load(str(Path(__file__).parent / "random_forest_data.npy")).mean(axis=3)
    scores = np.swapaxes(data, 1, 2)

    x_ax, y_ax, z_ax = np.meshgrid(x, y, z)

    kw = {
        'vmin': 0,
        'vmax': np.max(scores),
        'levels': np.linspace(0, np.max(scores), 20),
    }

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    contour = ax.contourf(
        x_ax[:, :, 0], y_ax[:, :, 0], scores[:, :, -1],
        zdir='z', offset=z_ax.max(), **kw
    )
    _ = ax.contourf(
        x_ax[0, :, :], scores[0, :, :], z_ax[0, :, :],
        zdir='y', offset=y_ax.min(), **kw
    )
    _ = ax.contourf(
        scores[:, -1, :], y_ax[:, -1, :], z_ax[:, -1, :],
        zdir='x', offset=x_ax.max(), **kw
    )

    x_min, x_max = x_ax.min(), x_ax.max()
    y_min, y_max = y_ax.min(), y_ax.max()
    z_min, z_max = z_ax.min(), z_ax.max()

    if selected_depth is not None:
        ax.plot([x_min, x_max], [y_min, y_min], [selected_depth, selected_depth], color="red",
                linewidth=2, zorder=2e3)
        ax.plot([x_max, x_max], [y_min, y_max], [selected_depth, selected_depth], color="red",
                linewidth=2, zorder=2e3)

    edges_kw = dict(color='gray', linewidth=1, zorder=1e3)
    ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], **edges_kw)
    ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], **edges_kw)
    ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], **edges_kw)

    ax.set(
        xlim=[x_min, x_max],
        xticks=x,
        xlabel=r"Number of trees per forest",
        xticklabels=2 ** x,
        ylim=[y_min, y_max],
        yticks=y,
        yticklabels=2 ** y,
        ylabel=r"Size of training subsets",
        zlim=[z_min, z_max],
        zticks=z,
        zticklabels=2 ** z,
        zlabel=r"Maximum tree depth",
        title="Accuracy score by the number of trees per forest,\n"
              "the size of the training subsets\n"
              "and the maximum tree depth ",
    )

    ax.view_init(30, -40, 0)
    ax.set_box_aspect(None, zoom=0.9)

    c_bar = fig.colorbar(mappable=contour, ax=ax, fraction=0.02, pad=0.1)
    c_bar.set_label('Accuracy Score')
    c_bar.set_ticks(np.arange(0, 1, 0.2))
    c_bar.set_ticklabels(np.arange(0, 1, 0.2).round(1))

    fig.savefig(f"accuracy_cube_depth_{selected_depth}.png", format="png", dpi=384)
    plt.show()


def mesure_accuracy_rf():
    t1 = time.thread_time_ns()
    emnist_train_x, emnist_train_y, emnist_test_x, emnist_test_y = split_dataset(1000, Emnist)
    pca = PCA(0.9)

    pca.fit(emnist_train_x)
    emnist_compressed_train_x = pca.compress(emnist_train_x)
    emnist_compressed_test_x = pca.compress(emnist_test_x)

    forest = RandomForest(tree_number=128, max_depth=8, training_subset_size=256, do_bagging=True,
                          class_number=10)
    sk_forest = RandomForestClassifier(n_estimators=128, max_depth=8,
                                       max_samples=256)
    forest.fit(emnist_compressed_train_x, emnist_train_y)
    sk_forest = sk_forest.fit(emnist_compressed_train_x, emnist_train_y)

    predicted_test_y = forest.predict(emnist_compressed_test_x)
    predicted_train_y = forest.predict(emnist_compressed_train_x)

    sk_predicted_test_y = sk_forest.predict(emnist_compressed_test_x)
    sk_predicted_train_y = sk_forest.predict(emnist_compressed_train_x)

    print(f"Score sur l'ensemble d'entraînement: "
          f"{accuracy_score(emnist_train_y, predicted_train_y):.3f}")
    print(f"Score sur l'ensemble de test : {accuracy_score(emnist_test_y, predicted_test_y):.3f}")
    print(f"Score sur l'ensemble d'entraînement: "
          f"{accuracy_score(emnist_train_y, sk_predicted_train_y):.3f} (sklearn)")
    print(f"Score sur l'ensemble de test :"
          f" {accuracy_score(emnist_test_y, sk_predicted_test_y):.3f} (sklearn)")

    test_conf_mat = confusion_matrix(Emnist.class_number, emnist_test_y, predicted_test_y)
    sk_test_conf_mat = confusion_matrix(Emnist.class_number, emnist_test_y, sk_predicted_test_y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title("Normalised Confusion Matrix (ias module)")
    show_confusion_matrix(test_conf_mat, Emnist.class_names, ax=ax1, round_decimal=2)

    ax2.set_title("Normalised Confusion Matrix (sklearn module)")
    show_confusion_matrix(sk_test_conf_mat, Emnist.class_names, ax=ax2, round_decimal=2)

    fig.savefig("conf_mats.png", format="png", dpi=384)
    fig.show()
    print(time.thread_time_ns() - t1)
