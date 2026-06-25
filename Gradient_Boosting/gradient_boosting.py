# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import display #display dataframes
from sklearn import metrics

from decision_tree import DecisionTree #importation of the DecisionTree class from the decision_tree module


# 2. Data Loading and Preparation
df = pd.read_csv("data/train.csv")
df = df.head(200) #we take only the first 200 rows to speed up the training of the model and the generation of the graphs


# 3. Display the dataset with all rows visible for debugging purposes
pd.set_option('display.max_rows', None)
print(df) #display the entire dataset to check for any issues with the data loading process


# 4. Feature and target selection
features = ['Packet Length Mean'] #we only use the 'Packet Length' feature for our model, as it is the most relevant feature for predicting the 'Anomaly Scores' target variable based on our analysis of the dataset
target = 'Label' #the target variable we want to predict is the 'Anomaly Scores', which represents the level of anomaly in the network traffic and is crucial for our cybersecurity threat detection model


# 5. creation of the feature matrix X and the target vector y
x = df[features]
y = df[target]


# 6. Initialisation of the feature matrix and target vector for the Gradient Boosting algorithm
y = y.values[:,None]


# 7. Initialisation of Gradient Boosting variables
xi = x #xi stores the input features (Packet Length)

yi = y #yi stores the targets, which change into residuals at each step

ei = 0 #ei stores the residual error (Actual Y - Predicted Y)

n = len(yi) #n is the total number of rows (200 rows)

predf = 0 #predf stores the final combined prediction of the ensemble


# 8. Gradient Boosting Loop
for i in range(30): #Run 30 iterations to create an ensemble of 30 trees

    # tree creation
    tree = DecisionTree(xi, yi) #Create a new decision tree based on current features and residuals

    # best split
    tree.find_better_split(0) #Find the best value to split the data and minimize error

    # split position
    r = np.where(xi == tree.split)[0][0] #Find the row index where the split happens

    # left indices
    left_idx = np.where(xi <= tree.split)[0] #Get the row indexes for the left side of the split (<= split value)

    # right indices
    right_idx = np.where(xi > tree.split)[0] #Get the row indexes for the right side of the split (> split value)

    # current tree prediction
    predi = np.zeros(n) #Create an empty array of zeros for the current tree predictions

    # left prediction
    np.put(
        predi,
        left_idx,
        np.repeat(np.mean(yi[left_idx]), r)
    ) #Assign the average residual value of the left side to the left indices

    # right prediction
    np.put(
        predi,
        right_idx,
        np.repeat(np.mean(yi[right_idx]), n-r)
    ) #Assign the average residual value of the right side to the right indices

    # column transformation
    predi = predi[:,None] #Convert the flat 1D prediction array into a 2D column vector

    # add current prediction
    predf = predf + predi #Add the current tree's prediction to the final ensemble prediction

    # calculate residuals
    ei = y - predf #Calculate the new error residuals (True Y - Predicted Y)

    # update residuals
    yi = ei #Set the targets for the next tree to be these new residuals


# 9. Live Prediction Function
def predict_new_data(new_x): #Function to predict anomaly scores for new incoming packets

    prediction = 0 #Initialize the prediction score to 0

    for i in range(len(new_x)): #Loop through each new data row passed in

        prediction += np.mean(predf) #Add the average trained ensemble prediction to the score

    return prediction #Return the final numerical anomaly score prediction