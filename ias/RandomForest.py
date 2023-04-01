import numpy

from . import DecisionTree


class RandomForest:
    def __init__(self, trees_number, subset_size, **args):
        self._subset_size = subset_size
        self._trees = []

        for _ in range(trees_number):
            self._trees.append(DecisionTree(**args))

        self._fitted = False

    def _check_for_fit(self):
        if not self._fitted:
            raise RuntimeError("RandomForest must be fitted")

    def fit(self, x, y) -> None:
        """ Crée des arbres avec les données labellisées (x, y) """
        indices = numpy.arange(len(x))

        for tree in self._trees:
            subset_indices = numpy.random.choice(indices, size=self._subset_size, replace=True)
            tree.fit(x[subset_indices], y[subset_indices])

    def predict(self, x) -> "y like":
        """ Prend des données non labellisées puis renvoi les labels estimés """
        self._check_for_fit()
        return numpy.argmax(self.predict_proba(x), axis=1).reshape(-1, 1)

    def predict_proba(self, x) -> "proba of class x for each elm":
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        self._check_for_fit()

        proba_sum = None
        for tree in self._trees:
            if proba_sum:
                proba_sum += tree.predict_proba(x)
            else:
                proba_sum = tree.predict_proba(x)
        proba_sum /= len(self._trees)
        return proba_sum
