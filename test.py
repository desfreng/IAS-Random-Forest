
import numpy
#from ias import RandomForest

csv = numpy.genfromtxt ('iris.data.csv', delimiter=",")
x = csv[:, :-1]
y = csv[:, [-1]]

length = len(x)
indices = numpy.arange(length)
numpy.random.shuffle(indices)
train_size = int(length*2/3)

train_i = indices[:train_size]
train_x = x[train_i]
train_y = y[train_i]

check_i = indices[train_size:]
check_x = x[check_i]
check_y = y[check_i]

#forest.fit(x, y, 250, 50)