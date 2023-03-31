import numpy as np


class PCA:
    def __init__(self, n_composantes: float | int):
        if isinstance(n_composantes, float) and not (0 <= n_composantes <= 1):
            raise ValueError("n_composantes must be in [0; 1]")

        self._n_comp = n_composantes
        self._data_dim: None | int = None

        self._fitted = False

        self._transformation_matrix: None | np.ndarray = None
        self._average_vector: None | np.ndarray = None

    def _check_for_fit(self):
        if not self._fitted:
            raise RuntimeError("PCA reduction must be fitted")

    @staticmethod
    def _covariance_and_average(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nb_features = len(data[0])
        nb_data_points = len(data)

        data_average = data.sum(axis=0) / nb_data_points
        cov_matrix = np.zeros((nb_features, nb_features))

        for data_point in range(nb_data_points):
            centered_data_vector = data[data_point] - data_average
            cov_matrix += centered_data_vector[np.newaxis].T @ centered_data_vector[np.newaxis]

        return cov_matrix / nb_data_points, data_average

    def fit(self, input_data: np.ndarray) -> None:
        self._data_dim = len(input_data[0])

        covariance_matrix, self._average_vector = self._covariance_and_average(input_data)

        # eigen_values (and vectors) are returned in ascending order !
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        # Find the number of vector needed if a variance ratio is given
        if isinstance(self._n_comp, float):
            target_variance = self._n_comp * np.sum(eigen_values)
            conserved_variance = 0
            eigen_index = 0

            while conserved_variance < target_variance and eigen_index < self._data_dim:
                conserved_variance += eigen_values[-eigen_index]
                eigen_index += 1

            self._n_comp = eigen_index

        self._transformation_matrix = eigen_vectors[:, -self._n_comp:]
        self._fitted = True

    def compress(self, input_data: np.ndarray) -> np.ndarray:
        self._check_for_fit()
        return np.array([(vector - self._average_vector) @ self._transformation_matrix
                         for vector in input_data])

    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        self._check_for_fit()
        return np.array([(vector @ self._transformation_matrix.T) + self._average_vector
                         for vector in compressed_data])

    @property
    def output_dimension(self):
        self._check_for_fit()
        return self._n_comp

    @property
    def input_dimension(self):
        self._check_for_fit()
        return self._data_dim
