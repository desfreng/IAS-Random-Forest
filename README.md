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

- Decision Tree
L'algorithme de random forest se base sur celui-là. Le Decision Tree, est une méthode d'apprentissage pour la classification et la régression. Ici nous avons que traité le cas de la classification. Le concept est simple: c'est un arbre binaire avec à chaque noeud un seuil correspondant à une feature. Ce seuil permet de discriminer les éléments qui se séparent alors. Aux feuilles, on retrouve donc des données classées avec une probabilité par classe. Pour prédire il suffit alors de regarder dans quelle feuille tompe l'élément, et puis de prendre la probabilité la plus haute. Le Decision Tree est donc simple à comprendre (on comprend exactement le raisonnement derrière chaque classification), à visualiser et demande peu de pré-traitement pour les données. Il est en plus rapide pour prédire, une qualité lorsque l'objectif du projet est d'en créer toute une forêt ! Mais cette méthode a aussi des inconvénients, comme une tendance accrue à overfitter. Cela dépend aussi de la condition d'arrêt lors de la création d'arbre: mettre une profondeur limite peut changer la donne. L'implémentation naïve est aussi assez instable dans le sens où des petits changement de training set peuvent générer des arbres totalement différents. Mais ces problèmes sont atténués lorsqu'on utilise l'approche Random Forest, en multipliant les arbres, on évite les cas limites. Dans notre implémentation, nous sommes allés plus loin que la version naïve du Decision Tree. Nous avons implémenté l'algorithme CART ainsi que du Bagging pour les features. En effet, à chaque noeud pour trouver le seuil discriminant, on ne va s'intéresser qu'à un sous ensemble des features, tiré avec remise. Cette création artificielle de biais permet de réduire la variance ainsi que de réduire la compléxité. Pour trouver le meilleur seuil, il faut une mesure de la qualité de la séparation et nous en avons implémenté deux: le Gini criterion ainsi que le Log-loss criterion. Dans la partie test, on compare les deux solutions. Enfin, nous avons travaillé sur la construction de Random Decision Tree dont les seuils de discrimination sont tirés au hasard. Cette algorithme sert pour le "Extremely Randomized Forest" qu'on aborde juste en bas.

Random Forest, les avantages et inconvénients, description du travail (options etc...)

## L'heure des tests sur des vrais datasets !

## Nos conclusions sur notre travail et des pistes futures :
- Pre-pruning et pruning (si on ne le fait pas)

## Ressources (on garde ?)

Nous nous sommes basé sur la structure de sklearn et leur doc nous a grandement aidé.
 - https://scikit-learn.org/stable/modules/tree.html
 - https://en.wikipedia.org/wiki/Decision_tree
 - https://towardsmachinelearning.org/decision-tree-algorithm/
 - https://www.ibm.com/topics/decision-trees
 - https://medium.com/geekculture/decision-trees-with-cart-algorithm-7e179acee8ff
 - https://medium.com/@pralhad2481/chapter-3-decision-tree-learning-part-2-issues-in-decision-tree-learning-babdfdf15ec3
