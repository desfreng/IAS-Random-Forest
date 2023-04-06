# Projet IAS-Random-Forest

Ce projet présente une implémentation personnalisée de l'algorithme de classification Random Forest, incluant plusieurs variantes et techniques de prétraitement des données. Le code est disponible sous la forme de notebooks Jupyter et de fichiers Markdown pour faciliter l'expérimentation et la compréhension des différentes étapes.

## Table des matières

1. [Prétraitement des données avec PCA](#1-prétraitement-des-données-avec-pca)
2. [Decision Tree Classifier](#2-decision-tree-classifier)
3. [Bagging](#3-bagging)
4. [Random Forest Classifier](#4-random-forest-classifier)
5. [Random Splitter](#5-random-splitter)
6. [Pruning](#6-pruning)

### 1. Prétraitement des données avec PCA

Dans cette partie, nous abordons la réduction de dimensionnalité des attributs en utilisant la méthode PCA (Principal Component Analysis). Cela permet d'améliorer les performances de l'algorithme en réduisant le nombre de dimensions à traiter.

Notebook Jupyter: [docs/PCA.ipynb](docs/PCA.ipynb)

### 2. Decision Tree Classifier

Ici, nous présentons l'implémentation de l'algorithme CART (Classification and Regression Trees) pour les arbres de décision. Nous discutons également des avantages et inconvénients de cette méthode, ainsi que des critères de sélection des seuils (Gini et Log-loss).

Notebook Jupyter: [docs/DecisionTree.ipynb](docs/DecisionTree.ipynb)

### 3. Bagging

Dans cette section, nous explorons la technique de Bagging (Bootstrap Aggregating) qui permet de réduire la variance et d'améliorer la performance des arbres de décision en créant plusieurs arbres et en combinant leurs résultats.

Notebook Jupyter: [docs/Bagging.ipynb](docs/Bagging.ipynb)

### 4. Random Forest Classifier

Dans cette partie, nous implémentons l'algorithme Random Forest, qui repose sur le Decision Tree Classifier et le comparons à un simple Decision Tree Classifier avec la technique de Bagging.

Notebook Jupyter: [docs/RandomForest.ipynb](docs/RandomForest.ipynb)

### 5. Random Splitter

Pour aller plus loin nous présentons ici la variante Random Splitter, qui utilise des seuils de séparation tirés au hasard pour réduire le temps de calcul et la variance. Nous décrivons les différences par rapport à l'algorithme Random Forest et les avantages et inconvénients de cette approche.

Notebook Jupyter: [docs/RandomSplitter.ipynb](docs/RandomSplitter.ipynb)

### 6. Pruning

Enfin, nous abordons l'élagage des arbres de décision (pruning), c'était plus compliqué que prévu donc nous n'avons pas d'implémentation utilisable.

Fichier Markdown: [docs/Pruning.md](docs/Pruning.md)