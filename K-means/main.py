import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from kmeans import KMeans

# ── Load data ─────────────────────────────────────────────────────
train = pd.read_csv('../train.csv')
test  = pd.read_csv('../test.csv')

X_train = train.drop('Label', axis=1).values
X_test  = test.drop('Label', axis=1).values

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ── K-Means from scratch ──────────────────────────────────────────
k = KMeans(K=6, max_iters=100)
y_pred = k.predict(X_test)

# ── Visualization with PCA ────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='Set1', alpha=0.3, s=5)
plt.title("K-Means From Scratch — PCA 2D")
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