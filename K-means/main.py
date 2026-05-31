# -*- coding: utf-8 -*-
"""
main.py - K-Means from Scratch
Folder layout:
  AI_SIHAM/
    K-means/
      kmeans.py
      main.py   <- this file
    data/
      train.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# Make sibling kmeans.py importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from kmeans import KMeans

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Find data/train.csv
# ─────────────────────────────────────────────────────────────────────────────
def find_csv(filename):
    # THIS_DIR = AI_SIHAM/K-means
    # parent   = AI_SIHAM/
    parent = os.path.normpath(os.path.join(THIS_DIR, '..'))
    candidates = [
        os.path.join(parent, 'data', filename),   # AI_SIHAM/data/train.csv  <-- correct
        os.path.join(THIS_DIR, 'data', filename),  # K-means/data/train.csv
        os.path.join(THIS_DIR, filename),          # K-means/train.csv
        os.path.join(os.getcwd(), 'data', filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            print("[OK] Found: " + p)
            return p
    msg = "[ERROR] Could not find '" + filename + "'.\n   Tried:\n"
    msg += "\n".join("   - " + os.path.normpath(c) for c in candidates)
    raise FileNotFoundError(msg)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load & sample FIRST (before extracting X)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 5000   # change to 10000 if you want more, but 5000 is fast

print("Loading dataset ...")
train_path = find_csv('train.csv')
train = pd.read_csv(train_path)
print("Full dataset shape : " + str(train.shape))

# Sample BEFORE making X — this is the key fix
print("Sampling " + str(SAMPLE_SIZE) + " rows ...")
train = train.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

X = train.drop('Label', axis=1).values.astype(float)
print("X shape (sampled) : " + str(X.shape))
print("")

# ─────────────────────────────────────────────────────────────────────────────
# 3. K-Means from scratch
# ─────────────────────────────────────────────────────────────────────────────
K = 6
print("Running K-Means (K=" + str(K) + ", max_iters=100) ...")

k = KMeans(K=K, max_iters=100)
y_pred = k.predict(X)

print("[OK] K-Means done.")
print("Cluster counts : " + str(dict(zip(*np.unique(y_pred, return_counts=True)))))
print("")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA 2D Visualization
# ─────────────────────────────────────────────────────────────────────────────
print("Running PCA ...")
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y_pred, cmap='Set1',
    alpha=0.4, s=8
)
plt.colorbar(scatter, label='Cluster')
plt.title("K-Means From Scratch - PCA 2D (n=" + str(SAMPLE_SIZE) + ")")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()

out_png = os.path.join(THIS_DIR, 'kmeans_pca.png')
plt.savefig(out_png, dpi=150)
plt.show()
print("[OK] Plot saved -> " + out_png)
print("")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Silhouette Score
# ─────────────────────────────────────────────────────────────────────────────
print("Computing Silhouette Score ...")
score = silhouette_score(X, y_pred)   # no sample_size needed, X is already small
print("Silhouette Score : " + str(round(score, 4)))
print("")
print("K-Means from scratch - done.")
