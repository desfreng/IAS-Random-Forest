import numpy as np


class PCA:
    def __init__(self, n_composantes: float | int):
        if isinstance(n_composantes, float) and not (0 <= n_composantes <= 1):
            raise ValueError("n_composantes must be in [0; 1]")

        self._n_comp = n_composantes
        self._data_dim: None | int = None

        self._fitted = False

        self._compression_matrix: None | np.ndarray = None
        self._decompression_matrix: None | np.ndarray = None
        self._average_vector: None | np.ndarray = None

    def _check_for_fit(self):
        if not self._fitted:
            raise RuntimeError("PCA reduction must be fitted")

    def fit(self, input_data: np.ndarray) -> None:
        self._data_dim = len(input_data[0])

        covariance_matrix = input_data
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        permutation_map = np.argsort(eigen_values)

        sorted_eigen_values = eigen_values[permutation_map]
        sorted_eigen_vectors = eigen_vectors[:, permutation_map]

        selected_vectors: list[np.ndarray] = []

        if isinstance(self._n_comp, float):
            target_variance = self._n_comp * np.sum(eigen_values)
            conserved_variance = 0
            eigen_index = 0

            while conserved_variance < target_variance:
                selected_vectors.append(sorted_eigen_vectors[:, eigen_index])
                conserved_variance += sorted_eigen_values[eigen_index]
                eigen_index += 1

            self._n_comp = eigen_index
        elif isinstance(self._n_comp, int):
            selected_vectors.extend(sorted_eigen_values[: range(self._n_comp)])

        self._compression_matrix = np.empty((self._data_dim, self._n_comp))

        for index, vector in enumerate(selected_vectors):
            self._compression_matrix[:, index] = vector

        self._decompression_matrix = np.transpose(self._compression_matrix)
        self._fitted = True

    def compress(self, input_data: np.ndarray) -> np.ndarray:
        self._check_for_fit()
        return (input_data - self._average_vector) * self._compression_matrix

    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        self._check_for_fit()
        return compressed_data * self._decompression_matrix + self._average_vector

    @property
    def output_dimension(self):
        self._check_for_fit()
        return self._n_comp

    @property
    def input_dimension(self):
        self._check_for_fit()
        return self._data_dim