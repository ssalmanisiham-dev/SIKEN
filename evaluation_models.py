# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# 2. Loading the Dataset
df = pd.read_csv("data/train.csv")
df = df.sample(n=100000, random_state=42)

# 3. Overview of the Dataset
print("Dataset Overview:")
print(df.head())

# 4. Checking for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Defining Explanatory Variables (Features)
features = [
    'Bwd Packet Length Max', 'Packet Length Mean', 'Flow IAT Std', 'Active Min',
    'Active Mean', 'Flow Duration', 'Flow IAT Mean', 'Fwd IAT Mean',
    'Bwd Packet Length Min', 'ACK Flag Count', 'FIN Flag Count', 'Active Max',
    'Flow IAT Min', 'Min Packet Length', 'Bwd IAT Mean', 'Bwd IAT Min',
    'PSH Flag Count', 'Active Std', 'Bwd IAT Std', 'Fwd IAT Min'
]
target = "Label"

# 6. Cleaning Data
df_clean = df[features + [target]].dropna()

# 7. Creating X and y
X = df_clean[features]
y = df_clean[target]

# 8. Class Distribution
print("\nClass Distribution:")
print(y.value_counts())

# 9. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 10. Dimensions Information
print("\nDimensions:")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

# 11. Creating the Model Pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# 12. Training the Model
model.fit(X_train, y_train)

# 13. Predictions
y_pred = model.predict(X_test)

# 14. Accuracy Calculation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# 15. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 16. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 17. Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print("\nCross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# 18. Multiclass ROC AUC
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)
y_prob = model.predict_proba(X_test)
auc_score = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print("\nMulticlass ROC AUC:", auc_score)

# 19. Dashboard Visualization (4 plots)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Class Distribution
sns.countplot(x=y, ax=axes[0,0])
axes[0,0].set_title("Class Distribution")

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title("Confusion Matrix")

# Plot 3: Cross-Validation
axes[1,0].plot(range(1, len(scores)+1), scores, marker='o')
axes[1,0].set_title(f"Cross-Validation (Mean={scores.mean():.4f})")
axes[1,0].grid(True)

# Plot 4: ROC Multiclass
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    axes[1,1].plot(fpr, tpr, label=f"Class {classes[i]}")
axes[1,1].set_title(f"ROC Multiclass (AUC={auc_score:.4f})")
axes[1,1].legend()

plt.tight_layout()
plt.show()

# 20. Final Summary
print("\nFINAL SUMMARY")
print(f"Accuracy: {accuracy:.4f} | CV Mean: {scores.mean():.4f} | ROC AUC: {auc_score:.4f}")