import numpy as np
from scipy.io import loadmat


def emnist_digits_dataset(file_path):
    # Data downloaded from https://www.nist.gov/itl/products-and-services/emnist-dataset
    nb_class = 10  # Number of class

    digits = loadmat(file_path)['dataset'][0][0]
    train, test, _ = digits
    train = train[0][0]

    train_img, train_label, train_writer_id = train
    train_label = train_label[:, 0]

    def get_dist(nb_class, input_array):
        dist = np.empty(nb_class, dtype=int)

        for cls in range(nb_class):
            dist[cls] = np.count_nonzero(input_array == cls)

        return dist

    print(train_img.shape)
    print(get_dist(nb_class, train_label))

    np.savez_compressed("emnist-digits", label=train_label, data=train_img, nb_class=nb_class)


def iris_dataset(file_path: str):
    # Data downloaded from https://doi.org/10.24432/C56C76
    # Replace :
    #  -  "Iris-setosa"     with 0
    #  -  "Iris-versicolor" with 1
    #  -  "Iris-virginica"  with 2

    nb_class = 3  # Number of class

    data = np.genfromtxt(file_path, delimiter=",")
    features_data, label = data[:, :-1], data[:, -1]

    def get_dist(nb_class, input_array):
        dist = np.empty(nb_class, dtype=int)

        for cls in range(nb_class):
            dist[cls] = np.count_nonzero(input_array == cls)

        return dist

    print(features_data.shape)
    print(get_dist(nb_class, label))

    np.savez_compressed("iris", label=label, data=features_data, nb_class=nb_class)


emnist_digits_dataset("emnist-digits.mat")
iris_dataset("iris.data")
