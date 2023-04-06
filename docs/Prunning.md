```python
import numpy as np


# ----------------------------------- PRUNING -----------------------------------#

def compute_tree_score(nodes):
    """ Compute efficiency score from tree """
    return 1


def create_leaf_from_class(cls, samples, nb_cls):
    """ Create leaf dictionary from """
    proba_vector = np.zeros(nb_cls)
    proba_vector[cls] = 1
    leaf = {
        "is_node": False,
        "probability_vector": proba_vector,
        "samples": samples,
        "criterion": 1
    }
    return leaf


def acceptable(new_score, old_score):
    """Returns whether new_score is acceptable compared to old_score or not"""
    return new_score + 1e-4 >= old_score


def prune(self, node):
    """Remove inefficient nodes, simplifying the tree"""
    if not self._nodes[node]["is_node"]:
        return True
    left_son_id = self._nodes[node]["left_son_id"]
    right_son_id = self._nodes[node]["right_son_id"]
    pruned_left = self.prune(left_son_id)
    if pruned_left:
        pruned_right = self.prune(right_son_id)
        if pruned_right:
            # on 'prune' le nœud actuel
            cls, samples = self.compute_pruned_class(left_son_id, right_son_id)
            new_leaf = create_leaf_from_class(cls, samples, self._class_number)
            current_score = compute_tree_score(self._nodes)
            new_tree = self._nodes.copy()
            new_tree.pop(left_son_id)
            new_tree.pop(right_son_id)
            new_tree.pop(node)
            new_tree[node] = new_leaf
            new_score = compute_tree_score(new_tree)
            if acceptable(new_score, current_score):
                self._nodes = new_tree
                return True
            else:
                return False
    else:
        _ = self.prune(right_son_id)
        return False


def compute_pruned_class(self, left_son_id, right_son_id):
    """ Compute the more probable class for pruning
        Returns the most probable class ; number of samples
    """
    left_samples, left_p_vector = self._nodes[left_son_id]["samples"], self._nodes[left_son_id][
        "probability_vector"]
    right_samples, right_p_vector = self._nodes[right_son_id]["samples"], self._nodes[right_son_id][
        "probability_vector"]
    total_d_vector = left_samples * left_p_vector + right_samples * right_p_vector
    return np.argmax(total_d_vector), np.sum(total_d_vector)
```