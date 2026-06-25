import numpy as np
from iso_tree import build_tree, path_length, _avg_path_length


def isolation_forest(df, n_trees=100, max_depth=10, subspace=256):
    # Keep only numeric columns, convert to numpy array
    X = df.select_dtypes(include=[np.number]).values

    trees = []

    for i in range(n_trees):
        # Sample a fresh random subset of rows for this tree
        n_rows  = min(subspace, len(X))
        indices = np.random.choice(len(X), size=n_rows, replace=False)
        X_sample = X[indices]

        # Build one tree on this subset and save its root
        root = build_tree(X_sample, depth=0, max_depth=max_depth)
        trees.append(root)

    return trees   # e.g. a list of 100 tree roots



def anomaly_score(row_df, trees, subspace):
    # Convert the single row to a 1D numpy array of numbers
    x = row_df.select_dtypes(include=[np.number]).values.flatten()

    # Walk x through every tree and collect path lengths
    depths = [path_length(x, tree) for tree in trees]

    # Average path length across all trees
    avg_depth = np.mean(depths)

    # c = expected path length for the subspace size (normalisation constant)
    c = _avg_path_length(subspace)

    # Paper formula:  score = 2^(−avg_depth / c)
    score = 2 ** (-avg_depth / c)

    return score