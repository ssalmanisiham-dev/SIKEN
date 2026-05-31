# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .isolation_tree import *

# 2. Isolation Forest
def isolation_forest(df, n_trees=20, max_depth=10, subspace=256):
    forest = []

    for i in range(n_trees):
        # Sample the data subset based on the subspace parameter size
        sample_df = df.sample(frac=subspace) if subspace <= 1 else df.sample(subspace)
        
        # Create a new isolation tree and append it to our forest list
        tree = isolation_tree(sample_df, max_depth=max_depth)
        forest.append(tree)

    return forest


# 3. Path Length
def pathLength(example, iTree, path=0):
    path = path + 1
    question = list(iTree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # Route row down the left branch if true, else down the right branch
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
    else:
        answer = iTree[question][1]

    # Return depth path count if leaf is hit, else keep traversing recursively
    if not isinstance(answer, dict):
        return path
    else:
        return pathLength(example, answer, path=path)


# 4. Evaluate Instance
def evaluate_instance(instance, forest):
    # Collect path lengths from every tree inside the forest ensemble
    return [pathLength(instance, tree) for tree in forest]


# 5. c_factor
def c_factor(n):
    # Calculate the average path length of an unsuccessful search in a binary tree
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1.) / (n * 1.0))


# 6. Anomaly Score
def anomaly_score(data_point, forest, n):
    # Calculate mean depth and divide it by the normalization factor constant
    E = np.mean(evaluate_instance(data_point, forest))
    c = c_factor(n)
    return 2 ** -(E / c)


# 7. Instance Depth Plot
def instance_depth_plot(instance, outlier, forest):
    # Extract structural depth path metrics for comparison charts
    bars1 = evaluate_instance(outlier, forest)
    bars2 = evaluate_instance(instance, forest)

    # Set up graph tracking positions
    barWidth = 0.3
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Render normal and outlier comparison bar structures side by side
    plt.bar(r2, bars2, width=barWidth, label='Normal Sample')
    plt.bar(r1, bars1, width=barWidth, label='Outlier')

    # Label definitions
    plt.ylabel('Tree Depth')
    plt.xlabel('Trees')
    plt.title('Normal vs Outlier Path Length')
    plt.legend()
    plt.show()