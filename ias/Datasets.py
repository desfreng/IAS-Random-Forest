from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import requests
from scipy.io import loadmat

_EMNIST_URL = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
_IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

_emnist_file = Path("../datasets/emnist.npz")
_iris_file = Path("../datasets/iris.npz")


class Dataset:
    def __init__(self, dataset_name):
        if dataset_name == "enmist":
            self._data = self._setup_emnist()
        elif dataset_name == "iris":
            self._data = self._setup_iris()
        else:
            raise ValueError(f"dataset_name : {dataset_name} is not implemented")

    @property
    def labels(self) -> np.ndarray[int]:
        return self._data["labels"]

    @property
    def features(self) -> np.ndarray:
        return self._data["features"]

    @property
    def class_number(self) -> int:
        return self._data["class_number"]

    @property
    def class_mapping(self) -> list[str]:
        return list(map(str, self._data["class_mapping"]))

    @staticmethod
    def _setup_emnist():
        if not _emnist_file.exists():
            raise RuntimeError("EMNIST dataset not found. Please install it. "
                               "(see 'install_emnist' function")
        return np.load(str(_emnist_file))

    @staticmethod
    def _setup_iris():
        if not _emnist_file.exists():
            raise RuntimeError("IRIS dataset not found. Please install it. "
                               "(see 'install_iris' function")
        return np.load(str(_iris_file))


def install_iris():
    tmp_dir = TemporaryDirectory()
    class_mapping = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print("Installing IRIS... ")
    iris_data = requests.get(_IRIS_URL).content.decode()

    for class_id, class_name in enumerate(class_mapping):
        iris_data = iris_data.replace(class_name, str(class_id))

    downloaded_file = Path(tmp_dir.name) / Path(_IRIS_URL).name
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
    print(tmp_dir)

    print("Installing EMNIST, might take a while... ")
    downloaded_file = Path(tmp_dir.name) / Path(_EMNIST_URL).name
    with downloaded_file.open(mode="w+b") as f:
        f.write(requests.get(_EMNIST_URL).content)

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
