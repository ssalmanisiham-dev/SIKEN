# 1. Imports
import pandas as pd #data manipulation
import numpy as np #numerical computing
import matplotlib.pyplot as plt #visualization
import math #math functions
from IPython.display import display #display dataframes
from sklearn import metrics #evaluation metrics

from gradient_boosting import *


# 2. Data Dimensions
print("=== Data Dimensions ===")
print(x.shape, y.shape) #Print data rows and columns dimensions


# 3. Visualization Preparation
xa = np.array(x['Packet Length Mean']) #Convert feature column to a numpy array
order = np.argsort(xa) #Get sorted index positions for a smooth plot line
xs = np.array(xa)[order] #Sort packet lengths in ascending order
ys = np.array(predf)[order] #Sort predictions based on the packet length order


# 4. Create a Single Window with 3 Side-by-Side Plots
print("\n=== Generating Combined Plots ===")

# Create 1 row with 3 side-by-side graph sections (ax1, ax2, ax3)
f, (ax1, ax2, ax3) = plt.subplots(
    1, 
    3, 
    figsize=(18, 5) #Wide window size to fit everything cleanly
)


# --- Graph 1: Original Scatter Plot ---
ax1.plot(x, y, 'o', color='tab:blue') #Plot original data points in blue
ax1.set_title("1. Original Scatter Plot")
ax1.set_xlabel("Packet Length")
ax1.set_ylabel("Anomaly Scores")


# --- Graph 2: Gradient Boosting Prediction ---
ax2.plot(x, y, 'o', alpha=0.6) #Plot original scatter points with transparency
ax2.plot(xs, ys, 'r', linewidth=2) #Draw the red ensemble prediction line over data
ax2.set_title("2. Gradient Boosting Prediction")
ax2.set_xlabel("Packet Length")
ax2.set_ylabel("Anomaly Scores / y_pred")


# --- Graph 3: Residuals ---
ax3.plot(x, ei, 'go', alpha=0.6) #Plot residual errors (ei) as green points
ax3.axhline(y=0, color='black', linestyle='--') #Draw a dashed black center line at 0 error
ax3.set_title("3. Residuals vs Packet Length")
ax3.set_xlabel("Packet Length")
ax3.set_ylabel("Residuals")


# 5. Final Single Display
plt.tight_layout() #Fix overlapping titles and labels automatically
plt.show() #Display the final window with all 3 charts together


# Live Test

# Wrap the integer inside an array so it has a valid format and length
new_packet = np.array([500])

# Pass the test packet into the function to get a prediction score
prediction = predict_new_data(new_packet)

print("\n=== Live Prediction Test ===")
print(f"Packet Length checked : {new_packet[0]}") #Print the input packet length value
print(f"Predicted Anomaly Score: {prediction}") #Print the final anomaly score prediction