# 1. Imports
import pandas as pd #data manipulation
import numpy as np #numerical computing
import matplotlib.pyplot as plt #visualization
import math #math functions
from IPython.display import display #display dataframes
from sklearn import metrics #evaluation metrics


# 2. Function of standard deviation for aggregation
def std_agg(cnt, s1, s2):
    # Calculate variance, ensuring it doesn't drop below 0 due to floating point error
    variance = (s2 / cnt) - (s1 / cnt)**2
    return math.sqrt(max(0, variance))

# 3. Importation of Decision Tree
class DecisionTree(): #decision tree class


    # 4. Initialization of the decision tree
    def __init__(self, x, y, idxs=None, min_leaf=2):

        if idxs is None:
            idxs = np.arange(len(y)) #If no indexes are given, use all rows

        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf #Save input variables

        # number of samples and number of features
        self.n, self.c = len(idxs), x.shape[1] #Get number of rows (n) and columns (c)

        # value of the node (mean of the target values in the node)
        self.val = np.mean(y[idxs]) #Calculate the average target value for this node

        # score of the split (initialized to infinity)
        self.score = float('inf') #Set initial split score to infinity

        # find the best split
        self.find_varsplit() #Automatically start searching for the best split


    # 5. Search for the best split
    def find_varsplit(self):

        for i in range(self.c):
            self.find_better_split(i) #Check the split score for every feature column

        if self.is_leaf:
            return #Stop if this node cannot be split further (it's a leaf)

        x = self.split_col #Get the values of the chosen split column

        lhs = np.nonzero(x <= self.split)[0] #Find row indexes for the left child node
        rhs = np.nonzero(x > self.split)[0] #Find row indexes for the right child node

        # left child tree
        self.lhs = DecisionTree(
            self.x,
            self.y,
            self.idxs[lhs]
        ) #Create the left child branch recursively

        # right child tree
        self.rhs = DecisionTree(
            self.x,
            self.y,
            self.idxs[rhs]
        ) #Create the right child branch recursively


    # 6. Find better split for a given feature
    def find_better_split(self, var_idx):

        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs] #Get features and targets for active rows

        sort_idx = np.argsort(x) #Get sorted index positions for feature X

        sort_y, sort_x = y[sort_idx], x[sort_idx] #Sort both X and Y based on X values

        rhs_cnt, rhs_sum, rhs_sum2 = (
            self.n,
            sort_y.sum(),
            (sort_y**2).sum()
        ) #Start with all rows on the right side of the split

        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0. #Start with zero rows on the left side

        for i in range(0, self.n-self.min_leaf-1): #Loop through rows to test different split positions

            xi, yi = sort_x[i], sort_y[i] #Get current row's X feature and Y target

            lhs_cnt += 1 #Move one row counter to the left side
            rhs_cnt -= 1 #Remove one row counter from the right side

            lhs_sum += yi #Move Y value sum to the left
            rhs_sum -= yi #Remove Y value sum from the right

            lhs_sum2 += yi**2 #Move Y squared sum to the left
            rhs_sum2 -= yi**2 #Remove Y squared sum from the right

            if i < self.min_leaf or xi == sort_x[i+1]:
                continue #Skip if branch size is too small or if the next X value is identical

            lhs_std = std_agg(
                lhs_cnt,
                lhs_sum,
                lhs_sum2
            ) #Calculate standard deviation of left side

            rhs_std = std_agg(
                rhs_cnt,
                rhs_sum,
                rhs_sum2
            ) #Calculate standard deviation of right side

            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt #Calculate total split score (weighted error)

            if curr_score < self.score: #If current split has lower error than previous best:

                self.var_idx = var_idx #Update best feature index
                self.score = curr_score #Update best score value
                self.split = xi #Update best split threshold value


    # 7. Split column name
    @property 
    def split_name(self):
        return self.x.columns[self.var_idx] #Return the string name of the split column


    # 8. Split column values
    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx] #Return the data values of the split column


    # 9. Leaf node check
    @property
    def is_leaf(self):
        return self.score == float('inf') #It is a leaf if no valid split was found (score remains infinity)


    # 10. String representation of the tree
    def __repr__(self):

        s = f'n: {self.n}; val:{self.val}' #Create summary string with sample count and node mean value

        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}' #Add split info if not a leaf

        return s #Return tree description string


    # 11. Prediction for a set of samples
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x]) #Loop through rows to predict a whole dataset array


    # 12. Prediction for a single sample
    def predict_row(self, xi):

        if self.is_leaf:
            return self.val #If it's a leaf node, return its average value as the final prediction

        # Choose the left child branch if feature value <= split threshold, else choose right branch
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs

        return t.predict_row(xi) #Recursively traverse down the chosen branch until hitting a leaf