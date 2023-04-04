# Rapport IAS-Random-Forest

## Sur quoi nous avons travaillé :

Nous avons travaillé sur une implémentation de l'algorithme de classification Random Forest.
Notre travail met en œuvre chacune des étapes de la classification.
En effet, une première partie de notre travail porte sur le prétraitement des données.
Il fallait, pour pouvoir tester notre travail sur de vrais dataset, être capable de réduire en dimension les attributs.
Nous avons alors implémenté la PCA, avec chacune de ces étapes (à détailler Gaby).
Ensuite vient le cœur du programme, le Random Forest Classifier.
L'algorithme se base sur le Decision Tree Classifier, avec quelques variantes.
Nous avons donc implémenté l'algorithme CART ainsi qu'une version naïve pour pouvoir les comparer.
Nous avons voulu aller plus loin en implémentant l'algorithme "Extremely Randomized Tree" (pas sûr que ce soit le nom exact lol)
qui nécessite l'implémentation d'une version modifiée du Decision Tree Classifier.
Enfin, pour pouvoir comparer et tester nos algorithmes (entre eux ainsi que face aux versions de sklearn)
nous avons utilisé les dataset "Iris" et "eMNIST".
Pour pouvoir mieux visualiser les résultats, nous avons aussi travaillé sur un module de graphes
qui permet l'observation et l'exportation des beaux arbres créés !

(petit paragraphe sur la structure du code ?)

## Les algorithmes que nous avons implémentés :

PCA, pourquoi, les avantages et inconvénients, description du travail (options etc...)

Decision Tree, pourquoi, les avantages et inconvénients, description du travail (options etc...)

Random Forest, les avantages et inconvénients, description du travail (options etc...)

## L'heure des tests sur des vrais datasets !

## Nos conclusions sur notre travail et des pistes futures :

## Ressources (on garde ?)

Nous nous sommes basé sur la structure de sklearn et leur doc nous a grandement aidé.
 - https://scikit-learn.org/stable/modules/tree.html
 - https://en.wikipedia.org/wiki/Decision_tree
 - https://towardsmachinelearning.org/decision-tree-algorithm/
 - https://www.ibm.com/topics/decision-trees
 - https://medium.com/geekculture/decision-trees-with-cart-algorithm-7e179acee8ff
