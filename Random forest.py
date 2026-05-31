import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Load data ─────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

X_train = train.drop('Label', axis=1)
X_test  = test.drop('Label', axis=1)

le = LabelEncoder()
y_train = le.fit_transform(train['Label'])
y_test  = le.transform(test['Label'])

print("Data loaded ✓")

# ── 1. Entraînement ───────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

# ── 2. Prédiction ─────────────────────────────────────────────────
y_pred = rf.predict(X_test)

# ── 3. Évaluation ─────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 4. Matrice de confusion ────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — Random Forest\nAccuracy : {accuracy*100:.2f}%")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()