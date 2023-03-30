
import numpy
from ias import RandomForest

csv = numpy.genfromtxt ('iris.data.csv', delimiter=",")
x = csv[:, :-1]
y = csv[:, [-1]]


#forest.fit(x, y, 250, 50)