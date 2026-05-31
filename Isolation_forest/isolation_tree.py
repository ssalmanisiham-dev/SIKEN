# 1. Imports
import numpy as np
import pandas as pd
import random


# 2. Select Random Feature
def select_feature(data):
    # Randomly choose a column name from the given dataframe
    return random.choice(data.columns)


# 3. Select Random Value
def select_value(data, feat):
    # Pick a random continuous threshold value between the feature min and max
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi - mini) * np.random.random() + mini


# 4. Split Data
def split_data(data, split_column, split_value):
    # Separate data into left group (less/equal) and right group (greater)
    data_below = data[data[split_column] <= split_value]
    data_above = data[data[split_column] > split_value]
    return data_below, data_above


# 5. Classify Data
def classify_data(data):
    # Handle base case scenario where the data split subset is empty
    if data.shape[0] == 0:
        return 0

    # Extract labels from the final column and count unique class occurrences
    label_column = data.values[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    # Identify and return the majority class classification label inside the node
    index = counts_unique_classes.argmax()
    return unique_classes[index]


# 6. Isolation Tree
def isolation_tree(data, counter=0, max_depth=50):
    # Stop condition: check if maximum depth limit or terminal sample size is reached
    if counter == max_depth or data.shape[0] <= 1:
        return classify_data(data)

    else:
        counter += 1

        # Select a random splitting criteria layout
        split_column = select_feature(data)
        split_value = select_value(data, split_column)

        # Divide data points into left and right subsets
        data_below, data_above = split_data(data, split_column, split_value)

        # Initialize the current node's split question structure
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}

        # Run recursion loops to continue branching out down the tree paths
        below_answer = isolation_tree(data_below, counter, max_depth=max_depth)
        above_answer = isolation_tree(data_above, counter, max_depth=max_depth)

        # Collapse node if branches contain duplicate values, else append answers
        if below_answer == above_answer:
            sub_tree = below_answer
        else:
            sub_tree[question].append(below_answer)
            sub_tree[question].append(above_answer)

        return sub_tree