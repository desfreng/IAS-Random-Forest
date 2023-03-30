from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import requests
from scipy.io import loadmat

EMNIST_URL = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

_emnist_file = Path("./emnist.npz")
_iris_file = Path("./iris.npz")


def install_iris():
    tmp_dir = TemporaryDirectory()
    class_mapping = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print("Installing IRIS... ")
    iris_data = requests.get(IRIS_URL).content.decode()

    for class_id, class_name in enumerate(class_mapping):
        iris_data = iris_data.replace(class_name, str(class_id))

    downloaded_file = Path(tmp_dir.name) / Path(IRIS_URL).name
    with downloaded_file.open(mode="w") as f:
        f.write(iris_data)

    print("Downloaded!")
    print("Extracting Data...")
    nb_class = 3  # Number of class

    data = np.genfromtxt(downloaded_file, delimiter=",")
    features_data, label = data[:, :-1], data[:, -1]

    np.savez_compressed(_iris_file, labels=label, features=features_data,
                        class_number=nb_class, class_mapping=class_mapping)
    print("Done.")


def install_emnist():
    tmp_dir = TemporaryDirectory()

    print("Installing EMNIST, might take a while... ")
    downloaded_file = Path(tmp_dir.name) / Path(EMNIST_URL).name
    with downloaded_file.open(mode="w+b") as f:
        f.write(requests.get(EMNIST_URL).content)

    print("Downloaded!")
    print("Unzipping...")
    with ZipFile(downloaded_file) as tmp_zip:
        tmp_mat = tmp_zip.extract("matlab/emnist-digits.mat", tmp_dir.name)

    print("Extracting Data...")
    nb_class = 10  # Number of class

    digits_data = loadmat(tmp_mat)['dataset'][0][0]
    train, test, _ = digits_data
    train = train[0][0]

    train_img, train_label, train_writer_id = train
    train_label = train_label[:, 0]

    np.savez_compressed(_emnist_file, labels=train_label, features=train_img,
                        class_number=nb_class, class_mapping=list(map(str, range(nb_class))))
    print("Done.")


if __name__ == '__main__':
    install_iris()
    install_emnist()
