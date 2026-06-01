import numpy as np
from collections import Counter
from decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    if len(y) == 0:
        return 0
    counter = Counter(y)
    return counter.most_common(1)[0][0]


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2,
                 max_depth=10, n_feats=None):
        self.n_trees           = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth
        self.n_feats           = n_feats
        self.trees             = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"  Training tree {i+1}/{self.n_trees}...")
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)