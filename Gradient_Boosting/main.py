
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gradient_boosting import *

print("=== Data Dimensions ===")
print(x.shape, y.shape)

xa = np.array(x['Packet Length Mean'])

order = np.argsort(xa)

xs = xa[order]
ys = np.array(predf)[order]

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    y,
    'o',
    alpha=0.6,
    label='Real Data'
)

plt.plot(
    xs,
    ys,
    'r',
    linewidth=2,
    label='Gradient Boosting Prediction'
)

plt.title("Gradient Boosting Prediction")
plt.xlabel("Packet Length Mean")
plt.ylabel("Anomaly Score / Prediction")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

y_true = np.array(y)
y_pred = np.round(predf).astype(int)

print("\n===== Gradient Boosting Evaluation =====")

print("Accuracy  :", accuracy_score(y_true, y_pred))
print("Precision :", precision_score(
    y_true,
    y_pred,
    average='weighted',
    zero_division=0
))
print("Recall    :", recall_score(y_true, y_pred, average='weighted'))
print("F1-score  :", f1_score(y_true, y_pred, average='weighted'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
