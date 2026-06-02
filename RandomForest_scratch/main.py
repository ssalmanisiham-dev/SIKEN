import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from random_forest import RandomForest

# ── Load data ─────────────────────────────────────────────────────
train = pd.read_csv(r'data\train.csv')
test  = pd.read_csv(r'data\test.csv')

# Use sample for speed (from scratch is slow)
train = train.sample(n=5000, random_state=42)
test  = test.sample(n=1000, random_state=42)

X_train = train.drop('Label', axis=1).values
X_test  = test.drop('Label', axis=1).values

le = LabelEncoder()
y_train = le.fit_transform(train['Label'])
y_test  = le.transform(test['Label'])

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Classes : {le.classes_}")

# ── Train ──────────────────────────────────────────────────────────
print("\nTraining Random Forest from scratch...")
rf = RandomForest(n_trees=10, max_depth=10, n_feats=5)
rf.fit(X_train, y_train)

# ── Predict ────────────────────────────────────────────────────────
print("Predicting...")
y_pred = rf.predict(X_test)

# ── Evaluation ────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report :")
labels_present = np.unique(np.concatenate([y_test, y_pred]))
class_names = [str(name) for name in le.inverse_transform(labels_present)]
print(classification_report(y_test, y_pred, labels=labels_present, target_names=class_names))

# ── Confusion Matrix ──────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=labels_present)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — Random Forest From Scratch\nAccuracy : {accuracy*100:.2f}%")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nRandom Forest From Scratch terminé (Done)")
