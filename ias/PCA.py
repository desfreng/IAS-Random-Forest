
class PCA:
    def __init__(self, n_composantes: float | int):
        if isinstance(n_composantes, float):
            pass
        elif isinstance(n_composantes, int):
            self._n_comp = n_composantes

        pass

    def fit(self, x) -> None:
        pass

    def compress(self, x) -> "x reduced type":
        pass

    def decompress(self, y: "x reduced type") -> "x type":
        pass


