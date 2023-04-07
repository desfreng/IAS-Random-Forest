# Projet IAS-Random-Forest

Ce projet présente une implémentation personnalisée de l'algorithme de classification Random Forest,
incluant plusieurs variantes et techniques de prétraitement des données. Le code est disponible sous
la forme de notebooks Jupyter et de fichiers Markdown pour faciliter l'expérimentation et la
compréhension des différentes étapes.

## Table des matières

1. [Mise en Route](#1-mise-en-route)
2. [Datasets](#2-datasets)
3. [Prétraitement des données avec PCA](#3-prétraitement-des-données-avec-pca)
4. [Decision Tree Classifier](#4-decision-tree-classifier)
5. [Critères de Séparation](#5-critères-de-séparation)
6. [Bagging](#6-bagging)
7. [Random Forest Classifier](#7-random-forest-classifier)
8. [Random Splitter](#8-random-splitter)
9. [Pruning](#9-pruning)

### 1. Mise en Route

Le code de ce projet est rassemblé dans un module : `ias`. Celui-ci contient toutes les classes
nécessaires pour une utilisation des _Decision Tree_ ainsi que des _Random Forests_.
Pour exécuter le code, l'installation des dépendances ainsi que des datasets est nécessaire. Dans un
shell quelconque cela peut être réalisé de la manière suivante :

```shell
python3 -m venv ./venv  # On installe un environnement
source venv/bin/activate  # On rentre à l'intérieur
pip install -r requirements.txt  # Installation des dépendances (pip et pas conda !)
python3 datasets/install_datasets.py  # Installation des datasets (Long si connexion lente, EMNIST n'est pas léger)
```

Vous devriez ensuite être prêt à tout exécuter !

NB : Pour les Notebooks Jupyter, faire attention à lancer le serveur à la racine du projet pour
permettre l'importation du module `ias`.

### 2. Datasets

Une brève présentation des Datasets utilisés est disponible dans cette section. On expliquera
également l'utilisation et l'importation de ces derniers.

Notebook Jupyter: [docs/Datasets.ipynb](docs/Datasets.ipynb)

### 3. Prétraitement des données avec PCA

Dans cette partie, nous abordons la réduction de dimensionnalité des attributs en utilisant la
méthode PCA (Principal Component Analysis). Cela permet d'améliorer les performances de l'algorithme
en réduisant le nombre de dimensions à traiter.

Notebook Jupyter: [docs/PCA.ipynb](docs/PCA.ipynb)

### 4. Decision Tree Classifier

Ici, nous présentons l'implémentation de l'algorithme CART (Classification and Regression Trees)
pour les arbres de décision.

Notebook Jupyter: [docs/DecisionTree.ipynb](docs/DecisionTree.ipynb)

### 5. Critères de Séparation

Cette partie explore les différences entre les deux critères sélection des seuils implémentés :
_gini_ et _log loss_.

Notebook Jupyter: [docs/Criterion.ipynb](docs/Criterion.ipynb)

### 6. Bagging

Dans cette section, nous explorons la technique de Bagging (Bootstrap Aggregating) qui permet de
réduire la variance et d'améliorer la performance des arbres de décision en créant plusieurs arbres
et en combinant leurs résultats.

Notebook Jupyter: [docs/Bagging.ipynb](docs/Bagging.ipynb)

### 7. Random Forest Classifier

Dans cette partie, nous implémentons l'algorithme Random Forest, qui repose sur le Decision Tree
Classifier et le comparons à un simple Decision Tree Classifier avec la technique de Bagging.

Notebook Jupyter: [docs/RandomForest.ipynb](docs/RandomForest.ipynb)

### 8. Random Splitter

Pour aller plus loin, nous présentons ici la variante Random Splitter, qui utilise des seuils de
séparation tirés au hasard pour réduire le temps de calcul et la variance. Nous décrivons les
différences par rapport à l'algorithme Random Forest et les avantages et inconvénients de cette
approche.

Notebook Jupyter: [docs/RandomSplitter.ipynb](docs/RandomSplitter.ipynb)

### 9. Pruning

Enfin, nous abordons l'élagage des arbres de décision (pruning), c'était plus compliqué que prévu
donc nous n'avons pas d'implémentation utilisable.

Fichier Markdown: [docs/Pruning.md](docs/Pruning.md)