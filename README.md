# IAS-Random-Forest

À explorer:
- Decision trees
  - dimensionality reduction
    - PCA
    - Feature selection
  - tree depth depends on number of sample (x2 each add level)
  - vary the leafs and the split to control overfitting and learning
  - balance dataset (normalizing sum of the classes' weight)
  - CART algorithm
  - plot_tree ?


- RandomForest algorithm
  - each tree from a bootstrap sample (of training set)
  - best split : all features / random subset of features
  - randomness to decrease variance of the estimator
  - slight increase bias


- Extremely Randomized Trees method
  - instead of the most discriminative treshold, random treshold -> then we pick the best
  - again decrease variance and slightly increase bias

- multi-output problems (several outputs to predict instead of a single one)
  - first case : the n outputs are independant => n independant models
  - second case : correlation => single model and simultaneous prediction
  
- CART algorithm
  - Gini Index criterion