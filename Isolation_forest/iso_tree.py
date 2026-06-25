import numpy as np

class IsolationTreeNode:
    def __init__(self):
        self.feature   = None   # which column was cut  (e.g. 2  → column index 2)
        self.threshold = None   # the random cut value  (e.g. 3.47)
        self.left      = None   # child node for rows BELOW threshold
        self.right     = None   # child node for rows ABOVE threshold
        self.size      = 0      # number of data points at this node
        self.is_leaf   = False  # True when we stop splitting here



def build_tree(X, depth, max_depth):
    node      = IsolationTreeNode()
    node.size = len(X)

    # ── Stop conditions ──────────────────────────────────
    # Only 1 point left → nothing to split
    # OR we've reached the maximum allowed depth
    if len(X) <= 1 or depth >= max_depth:
        node.is_leaf = True
        return node

    # ── Pick a random feature and a random cut ───────────
    n_features = X.shape[1]
    feature    = np.random.randint(0, n_features)   # e.g. 2

    col_min = X[:, feature].min()
    col_max = X[:, feature].max()

    # All values are identical → can't split → leaf
    if col_min == col_max:
        node.is_leaf = True
        return node

    threshold = np.random.uniform(col_min, col_max)  # e.g. 3.47

    # ── Split and recurse ────────────────────────────────
    left_mask  = X[:, feature] < threshold
    right_mask = ~left_mask

    node.feature   = feature
    node.threshold = threshold
    node.left      = build_tree(X[left_mask],  depth + 1, max_depth)
    node.right     = build_tree(X[right_mask], depth + 1, max_depth)

    return node


def _avg_path_length(n):
    if n <= 1:
        return 0
    if n == 2:
        return 1
    # Euler–Mascheroni constant ≈ 0.5772
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


def path_length(x, node, current_depth=0):

    # Hit a leaf → add the expected extra depth for any remaining points
    if node.is_leaf:
        return current_depth + _avg_path_length(node.size)

    # Walk left or right depending on the cut
    if x[node.feature] < node.threshold:
        return path_length(x, node.left,  current_depth + 1)
    else:
        return path_length(x, node.right, current_depth + 1)