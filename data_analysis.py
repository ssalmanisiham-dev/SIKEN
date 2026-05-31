"""
=======================================================================
  DATA MINING - ANALYSE DU TRAFIC RÉSEAU (Dataset CVE)
  Cours : Data Analysis & Data Mining - ENSA de Kenitra
  Pr. Aniss MOUMEN
  
  Structure du projet (selon le PPT) :
  ├── ÉTAPE 1 : Chargement des données        (Slide 12-15)
  ├── ÉTAPE 2 : Compréhension des données     (Slide 99)
  ├── ÉTAPE 3 : Préparation / Nettoyage       (Slide 17-36)
  │   ├── Valeurs manquantes
  │   ├── Valeurs aberrantes (outliers)
  │   └── Visualisation avant / après
  ├── ÉTAPE 4 : Transformation                (Slide 23-28)
  │   ├── Encodage des labels
  │   └── Normalisation / Standardisation
  ├── ÉTAPE 5 : Feature Selection             (Slide 40-45)
  │   └── Corrélation + ANOVA
  ├── ÉTAPE 6 : Exploration / EDA             (Slide 46-53)
  │   └── Statistiques + Visualisations
  ├── ÉTAPE 7 : Modélisation Supervisée       (Slide 65-70)
  │   ├── Random Forest
  │
  ├── ÉTAPE 8 : Modélisation Non Supervisée   (Slide 90-93)
  │   └── Isolation Forest
  ├── ÉTAPE 9 : Modélisation Supervisée   (Slide 90-93)
  │   └── K-Means Clustering
 
=======================================================================
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score,silhouette_score, roc_auc_score, roc_curve
)
import seaborn as sns
warnings.filterwarnings('ignore')


# Style général des graphiques 
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#F5F5F5',
    'axes.edgecolor':   '#333333',
    'axes.grid':        True,
    'grid.color':       '#DDDDDD',
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})

# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv", low_memory=False)
df.columns = df.columns.str.strip()

# Uniformiser la colonne Label
if " Label" in df.columns:
    df.rename(columns={" Label": "Label"}, inplace=True)
df['Label'] = df['Label'].str.strip()

print(f"Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Types d'attaques : {df['Label'].nunique()}")
print(df['Label'].value_counts())


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — COMPRÉHENSION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════
print("  ÉTAPE 2 — COMPRÉHENSION DES DONNÉES")

print(df.shape)        # nombre de lignes et colonnes
print(df.head())       # 5 premières lignes
print(df.dtypes)       # types des colonnes
print(df['Label'].value_counts())  # distribution des classes

# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — PRÉPARATION DES DONNÉES 
# ═══════════════════════════════════════════════════════════════════
print("  ÉTAPE 3 — PRÉPARATION / NETTOYAGE")
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()

# ── 1. Remplacer inf par NaN ──────────────────────────────────────
df[cols_num] = df[cols_num].replace([np.inf, -np.inf], np.nan)
# ── 2. CALCULER les valeurs manquantes ───────────────────────────
print("Valeurs manquantes par colonne :")
print(df[cols_num].isnull().sum())

# ── 3. CALCULER les valeurs aberrantes (IQR) ─────────────────────
print("\nValeurs aberrantes par colonne :")
for col in cols_num:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    nb_outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col} : {nb_outliers}")



# ── 5. TRAITER — valeurs manquantes (médiane) ─────────────────────
for col in cols_num:
    df[col] = df[col].fillna(df[col].median())

# ── 6. TRAITER — outliers (écrêtage IQR) ─────────────────────────
# Suppression des outliers détectés (IQR)
Q1 = df[cols_num].quantile(0.25)
Q3 = df[cols_num].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[~((df[cols_num] >= lower_bound) & (df[cols_num] <= upper_bound)).all(axis=1)]
df_cleaned = df[((df[cols_num] >= lower_bound) & (df[cols_num] <= upper_bound)).all(axis=1)]

# Aperçu des tailles
print("Taille initiale :", df.shape[0])
print("Outliers détectés (IQR) :", outliers_iqr.shape[0])
print("Taille après nettoyage :", df_cleaned.shape[0])

# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — TRANSFORMATION DES DONNÉES
# ( encodage, normalisation, standardisation)
# ═══════════════════════════════════════════════════════════════════
print("  ÉTAPE 4 — TRANSFORMATION")
# ── 1 Encodage du Label (variable cible) ────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df['Label'])
print(f"\n  Encodage Label : {le.classes_.tolist()[:5]} ... → [0, 1, 2, ...]")

# ── Features X ────────────────────────────────────────────────
X = df[cols_num].copy()

# ── 2. StandardScaler ─────────────────────────────────
scaler_std = StandardScaler()
X_standard = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns)

# ── 3. RobustScaler ───────────────────────────────────
scaler_rob = RobustScaler()
X_robust = pd.DataFrame(scaler_rob.fit_transform(X), columns=X.columns)

# ── 4. Log Transformation ─────────────────────────────
X_log = X.copy()
for col in cols_num:
    if X[col].min() >= 0:
        X_log[col] = np.log1p(X[col])

# ── Visualisation comparaison ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
col_demo = cols_num[0]

axes[0].hist(X[col_demo], bins=50, color='#E74C3C')
axes[0].set_title("Original")

axes[1].hist(X_standard[col_demo], bins=50, color='#3498DB')
axes[1].set_title("StandardScaler")

axes[2].hist(X_robust[col_demo], bins=50, color='#2ECC71')
axes[2].set_title("RobustScaler")

axes[3].hist(X_log[col_demo], bins=50, color='#9B59B6')
axes[3].set_title("Log Transformation")

plt.suptitle("Comparaison des transformations", fontweight='bold')
plt.tight_layout()
plt.show()
# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 5 — SÉLECTION DES FEATURES
# ═══════════════════════════════════════════════════════════════════
print("ÉTAPE 5 — FEATURE SELECTION")

# ── 1. Matrice de corrélation (Slide 41-43) ────────────────────────
corr = X[cols_num[:15]].corr()

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(corr, annot=True, fmt=".1f", cmap='RdYlGn', center=0, linewidths=0.5)
ax.set_title("Matrice de corrélation")
plt.tight_layout()
plt.show()

# ── 2. Supprimer colonnes très corrélées (> 0.95) ─────────────────
corr_matrix  = X.corr().abs()
upper        = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_sel        = X.drop(columns=cols_to_drop)
print(f"Colonnes supprimées : {len(cols_to_drop)}")
print(f"Features restantes  : {X_sel.shape[1]}")

# ── 3. ANOVA F-score  ────────────────────────────────
f_scores, p_values = f_classif(X_sel, y)
importance_df = pd.DataFrame({
    'feature': X_sel.columns,
    'f_score': f_scores
}).sort_values('f_score', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['feature'][::-1], importance_df['f_score'][::-1], color='#9B59B6')
ax.set_title("Top 20 features — ANOVA F-score")
plt.tight_layout()
plt.show()

# ── 4. Garder top 20 features ─────────────────────────────────────
top_features = importance_df['feature'].tolist()
X_final      = X_sel[top_features]
print(f"Features finales : {X_final.shape[1]}")

# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 6 — EXPLORATION DES DONNÉES (EDA)
# ═══════════════════════════════════════════════════════════════════
print("ÉTAPE 6 — EDA")

# ── 1. Distribution des classes ───────────────────────────────────
class_counts = df['Label'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Barplot
axes[0].bar(class_counts.index, class_counts.values, color='steelblue')
axes[0].set_title("Distribution des classes")
axes[0].set_xlabel("Classe")
axes[0].set_ylabel("Nombre")
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
axes[1].set_title("Répartition des classes")

plt.tight_layout()
plt.show()

# ── 2. Histogrammes top features ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Distribution des 8 meilleures features")

for i, col in enumerate(top_features[:8]):
    ax = axes[i//4][i%4]
    ax.hist(X_final[col], bins=40, color='#2980B9')
    ax.set_title(col[:18], fontsize=8)

plt.tight_layout()
plt.show()
# No need to resample — just tell the model to pay more attention to minority classes
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

print("EDA terminée ✓")
# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 7 — train and Test 
# ═══════════════════════════════════════════════════════════════════

# Split 80% / 20% (Slide 103)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y,
    test_size=0.20,
    random_state=42
)

# Visualisation 
print("Dataset complet :", X_final.shape[0], "lignes")
print("─" * 40)
print("Train (80%)     :", X_train.shape[0], "lignes")
print("Test  (20%)     :", X_test.shape[0], "lignes")
print("─" * 40)
print("Features        :", X_train.shape[1], "colonnes")

# Sauvegarder les 2 fichiers
train = X_train.copy()
train['Label'] = y_train
test = X_test.copy()
test['Label'] = y_test

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
print("\n✅ train.csv et test.csv sauvegardés !")

# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 8— RANDOM FOREST CLASSIFICATION 
# ═══════════════════════════════════════════════════════════════════
print("ÉTAPE 8 — RANDOM FOREST")


# ── 1. Entraînement ───────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,        # 100 arbres
    class_weight='balanced', # fix class imbalance (Heartbleed)
    random_state=42
)
rf.fit(X_train, y_train)

# ── 2. Prédiction ─────────────────────────────────────────────────
y_pred = rf.predict(X_test)

# ── 3. Évaluation (Slide 107-110) ─────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 4. Matrice de confusion (Slide 107-108) ────────────────────────
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

# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 9 — K-MEANS CLUSTERING (Slide 81-89)
# ═══════════════════════════════════════════════════════════════════
print("ÉTAPE 9 — K-MEANS CLUSTERING")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

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

# ── 3. AVANT PCA — visualisation brute ───────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
            c=y_pred_kmeans, cmap='Set1', alpha=0.3, s=5)
plt.title("K-Means Clustering — AVANT PCA (Feature 1 vs Feature 2)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# ── 4. APRÈS PCA — visualisation 2D ──────────────────────────────
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

# ── 5. Évaluation complète ────────────────────────────────────────
score = silhouette_score(X_test, y_pred_kmeans, sample_size=10000)

y_test_encoded = LabelEncoder().fit_transform(test['Label'])
ari = adjusted_rand_score(y_test_encoded, y_pred_kmeans)
nmi = normalized_mutual_info_score(y_test_encoded, y_pred_kmeans)

print(f"\nSilhouette Score       : {score:.4f}")
print(f"Adjusted Rand Index    : {ari:.4f}")
print(f"Normalized Mutual Info : {nmi:.4f}")
print("\nK-Means terminé ✓")
