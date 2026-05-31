import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ── 1. Load data ───────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

X_train = train.drop('Label', axis=1)
X_test  = test.drop('Label', axis=1)

print("Data loaded ✓")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

# ── Load data ─────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

X_train = train.drop('Label', axis=1)
X_test  = test.drop('Label', axis=1)

print("Data loaded ✓")

# ── 1. Elbow Method ───────────────────────────────────────────────
inertias = []
k_values = range(2, 8)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o', color='steelblue')
plt.title("Elbow Method — Choix du k optimal")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.tight_layout()
plt.show()

# ── 2. Entraînement k=6 ───────────────────────────────────────────
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X_train)
y_pred_kmeans = kmeans.predict(X_test)

# ── 3. AVANT PCA ──────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
            c=y_pred_kmeans, cmap='Set1', alpha=0.3, s=5)
plt.title("K-Means Clustering — AVANT PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# ── 4. APRÈS PCA ──────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=y_pred_kmeans, cmap='Set1', alpha=0.3, s=5)
plt.title("K-Means Clustering — APRÈS PCA (2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

# ── 5. Évaluation ─────────────────────────────────────────────────
score = silhouette_score(X_test, y_pred_kmeans, sample_size=10000)
y_test_encoded = LabelEncoder().fit_transform(test['Label'])
ari = adjusted_rand_score(y_test_encoded, y_pred_kmeans)
nmi = normalized_mutual_info_score(y_test_encoded, y_pred_kmeans)

print(f"\nSilhouette Score       : {score:.4f}")
print(f"Adjusted Rand Index    : {ari:.4f}")
print(f"Normalized Mutual Info : {nmi:.4f}")
print("\nK-Means terminé ✓")