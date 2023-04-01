from pathlib import Path

import numpy as np

_dataset_dir = Path(__file__).parent.parent / "datasets"
_emnist_file = _dataset_dir / "emnist.npz"
_iris_file = _dataset_dir / "iris.npz"


class Dataset:
    def __init__(self, dataset_name):
        if dataset_name == "emnist":
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
                               "(see 'install_datasets' script)")
        return np.load(str(_emnist_file))

    @staticmethod
    def _setup_iris():
        if not _emnist_file.exists():
            raise RuntimeError("IRIS dataset not found. Please install it. "
                               "(see 'install_datasets' script)")
        return np.load(str(_iris_file))


Emnist = Dataset("emnist")
Iris = Dataset("iris")
