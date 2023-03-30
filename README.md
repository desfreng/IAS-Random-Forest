# IAS-Random-Forest

# Pour Louis :

```python
def _get_distribution(nb_class, input_array):
    dist = np.empty(nb_class, dtype=int)
    
    for cls in range(nb_class):
        dist[cls] = np.count_nonzero(input_array == cls)
        
    return dist
```

# Blablabla Recherche :

À explorer:
- Decision trees
  - dimensionality reduction
    - PCA
    - Feature selection
  - tree depth depends on number of sample (x2 each add level)
  - vary the leafs and the split to control overfitting and learning
  - balance dataset (normalizing sum of the classes' weight)
  - CART algorithm
  - plot_tree
  - Bagging (select random subset of feature, each split)


- RandomForest algorithm
  - each tree from a bootstrap sample (of training set)
  - best split : all features / random subset of features
  - randomness to decrease variance of the estimator
  - slight increase bias


- Extremely Randomized Trees method
  - instead of the most discriminative threshold, random threshold -> then we pick the best
  - again decrease variance and slightly increase bias

- multi-output problems (several outputs to predict instead of a single one)
  - first case : the n outputs are independent => n independent models
  - second case : correlation => single model and simultaneous prediction
  
- CART algorithm
  - Gini Index criterion (GI)
    - Use row values as thresholds
    - iterate on every row (sorted), calculate newGiniIndex
    - select the best threshold as the lowest GI

# Ressources 
Pouet pouet des liens, pouet pouet :

 - https://scikit-learn.org/stable/modules/tree.html
 - https://en.wikipedia.org/wiki/Decision_tree
 - https://towardsmachinelearning.org/decision-tree-algorithm/
 - https://www.ibm.com/topics/decision-trees
 - https://medium.com/geekculture/decision-trees-with-cart-algorithm-7e179acee8ff
