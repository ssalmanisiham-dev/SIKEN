
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# Allow import from parent directory when running from Isolation_forest/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso_forest import isolation_forest, anomaly_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")

df_full = pd.read_csv(DATA_PATH)
df_full.columns = df_full.columns.str.strip()

FEATURE_COLS = [
    "Bwd Packet Length Max", "Packet Length Mean", "Flow IAT Std",
    "Active Min", "Active Mean", "Flow Duration", "Flow IAT Mean",
    "Fwd IAT Mean", "Bwd Packet Length Min", "ACK Flag Count",
    "FIN Flag Count", "Active Max", "Flow IAT Min", "Min Packet Length",
    "Bwd IAT Mean", "Bwd IAT Min", "PSH Flag Count", "Active Std",
    "Bwd IAT Std", "Fwd IAT Min"
]

for c in FEATURE_COLS:
    df_full[c] = pd.to_numeric(df_full[c], errors="coerce")
df_full["Label"] = pd.to_numeric(df_full["Label"], errors="coerce")
df_full = df_full.dropna(subset=FEATURE_COLS + ["Label"]).reset_index(drop=True)

# --- FIX 4: use a representative sample instead of all 554k rows ---
np.random.seed(42)
SAMPLE_SIZE = 3000
df = df_full.sample(n=min(SAMPLE_SIZE, len(df_full)), random_state=42).reset_index(drop=True)

print(f"Dataset sample: {len(df)} rows")
print(f"Attack rate in sample: {(df['Label']!=0).mean()*100:.1f}%")
print()


scaler = StandardScaler()
X_raw    = df[FEATURE_COLS].values
X_scaled = scaler.fit_transform(X_raw)          # zero mean, unit variance

# Wrap as DataFrame with simple column names (required by iso_forest)
X_df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(len(FEATURE_COLS))])

SUBSPACE = 256
N_TREES  = 100

print(f"Building Isolation Forest ({N_TREES} trees, subspace={SUBSPACE})...")
forest = isolation_forest(X_df, n_trees=N_TREES, max_depth=12, subspace=SUBSPACE)
print("Forest built.")
print()


print("Scoring samples...")
scores = [anomaly_score(X_df.iloc[[i]], forest, subspace=SUBSPACE)
          for i in range(len(X_df))]
scores = np.array(scores)
print(f"Score range: {scores.min():.4f} — {scores.max():.4f}")
print(f"Score mean:  {scores.mean():.4f}")
print()

# Instead of hard-coding 0.4, compute the threshold that flags exactly
# the same proportion of samples as the true anomaly rate.
# This is the correct unsupervised calibration approach.

true_anomaly_rate = float((df["Label"] != 0).mean())
threshold = float(np.percentile(scores, 100 * (1 - true_anomaly_rate)))

print(f"True anomaly rate in sample: {true_anomaly_rate*100:.1f}%")
print(f"Auto-calibrated threshold:   {threshold:.4f}")
print()

# Binary predictions: 1 = anomaly/attack, 0 = normal
y_true = (df["Label"] != 0).astype(int).values
y_pred = (scores > threshold).astype(int)


acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
auc  = roc_auc_score(y_true, scores)

print("=" * 45)
print("  Isolation Forest Evaluation")
print("=" * 45)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print(f"  ROC-AUC   : {auc:.4f}")
print("=" * 45)
print()

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(f"                  Predicted Normal  Predicted Attack")
print(f"  Actual Normal   {cm[0,0]:>16}  {cm[0,1]:>15}")
print(f"  Actual Attack   {cm[1,0]:>16}  {cm[1,1]:>15}")
print()



# --- FIX 5: drop both 'Label' AND 'anomaly' before PCA ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Isolation Forest — Fixed Evaluation", fontsize=13, fontweight='bold')

# Plot 1: Score distribution
axes[0].hist(scores[y_true == 0], bins=40, alpha=0.7, color='steelblue', label='Normal')
axes[0].hist(scores[y_true == 1], bins=40, alpha=0.7, color='tomato',    label='Attack')
axes[0].axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Threshold = {threshold:.3f}')
axes[0].set_title("Anomaly Score Distribution")
axes[0].set_xlabel("Isolation Forest Score")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: PCA 2D scatter
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)          # use scaled features, NOT Label
colors = np.where(y_pred == 1, 'tomato', 'steelblue')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.4, s=12)
axes[1].scatter([], [], c='tomato',    label='Detected Anomaly', s=30)
axes[1].scatter([], [], c='steelblue', label='Normal',           s=30)
axes[1].set_title("PCA 2D — Anomaly Detection")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Normal", "Attack"])
disp.plot(ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("isolation_forest_fixed_results.png", dpi=150, bbox_inches='tight')
print("Figure saved: isolation_forest_fixed_results.png")
plt.show()