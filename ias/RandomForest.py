import numpy
from . import DecisionTree

class RandomForest:
    def __init__(self):
        pass

    def fit(self, x, y, trees_number, subset_size) -> None:
        """ Crée des arbre avec les données labellisées (x, y) """

        self.trees = []
        indices = numpy.arange(len(x))

        for _ in range(trees_number):
            subset_indices = numpy.random.choice(indices, size=subset_size, replace=True)
            tree = DecisionTree()
            tree.fit(x[subset_indices], y[subset_indices])
            self.trees.append(tree)
        
    def predict(self, x)-> "y like":
        """ Prend des données non labellisées puis renvoi les labels estimés """
        return numpy.argmax(self.predict_proba(x), axis=1).reshape(-1, 1)

    def predict_proba(self, x) -> "proba of class x for each elm":
        """ Prend des données non labellisées puis renvoi la proba de chaque label """
        sum = None
        for tree in self.trees:
            if sum:
                sum += tree.predict_proba(x)
            else:
                sum = tree.predict_proba(x)
        sum /= len(self.trees)
        return sum
