import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from kmeans import KMeans

# ── Load data ─────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

X_train = train.drop('Label', axis=1).values
X_test  = test.drop('Label', axis=1).values

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ── K-Means from scratch ──────────────────────────────────────────
k = KMeans(K=6, max_iters=100)
print("Starting K-Means...")
y_pred = k.predict(X_test)
print("K-Means finished!")

# ── Elbow Method ──────────────────────────────────────────────────
print("Running Elbow Method...")
inertias = []
k_values = range(2, 8)

for k_val in k_values:
    km = KMeans(K=k_val, max_iters=100)
    preds = km.predict(X_test)
    # Calculate inertia manually
    inertia = 0
    for i in range(k_val):
        cluster_points = X_test[preds == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            inertia += ((cluster_points - centroid) ** 2).sum()
    inertias.append(inertia)
    print(f"  k={k_val} done")

plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o', color='steelblue')
plt.title("Elbow Method — Choix du k optimal")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.tight_layout()
plt.show()

# ── AVANT PCA — raw features ──────────────────────────────────────
k = KMeans(K=6, max_iters=100)
print("Starting K-Means...")
y_pred = k.predict(X_test)
print("K-Means finished!")

plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 0], X_test[:, 1],
            c=y_pred, cmap='Set1', alpha=0.3, s=5)
plt.title("K-Means From Scratch — AVANT PCA (Feature 1 vs Feature 2)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# ── APRÈS PCA ─────────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='Set1', alpha=0.3, s=5)
plt.title("K-Means From Scratch — APRÈS PCA (2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

# ── Evaluation ────────────────────────────────────────────────────
score = silhouette_score(X_test, y_pred, sample_size=5000)
y_encoded = LabelEncoder().fit_transform(test['Label'])
ari = adjusted_rand_score(y_encoded, y_pred)
nmi = normalized_mutual_info_score(y_encoded, y_pred)

print(f"\nSilhouette Score       : {score:.4f}")
print(f"Adjusted Rand Index    : {ari:.4f}")
print(f"Normalized Mutual Info : {nmi:.4f}")
print("\nK-Means From Scratch terminé ✓")