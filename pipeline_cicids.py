"""
CICIDS 2017 — Pipeline complet
================================================
1. Chargement des données
2. Exploration
3. Pré-traitement (nulls, doublons, outliers)
4. Encodage
5. Normalisation
6. Train / Test Split
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
print("\n=== 1. CHARGEMENT ===")

CSV_FOLDER = "MachineLearningCVE"
frames = []

for fname in sorted(os.listdir(CSV_FOLDER)):
    path = os.path.join(CSV_FOLDER, fname)
    df_tmp = pd.read_csv(path, low_memory=False)
    df_tmp.columns = df_tmp.columns.str.strip()
    frames.append(df_tmp)
    print(f"  {fname} → {len(df_tmp):,} lignes")

df = pd.concat(frames, ignore_index=True)
df.columns = df.columns.str.strip()

# Uniformiser le nom de la colonne cible
if " Label" in df.columns:
    df.rename(columns={" Label": "Label"}, inplace=True)

print(f"\nDataset combiné : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Types d'attaques : {df['Label'].nunique()}")

# ─────────────────────────────────────────
# 2. EXPLORATION
# ─────────────────────────────────────────
print("\n=== 2. EXPLORATION ===")

print("\n-- Aperçu des données --")
print(df.head())

print("\n-- Informations générales --")
print(df.info())

print("\n-- Statistiques descriptives --")
print(df.describe())

print("\n-- Distribution des classes (Label) --")
print(df["Label"].value_counts())

# ─────────────────────────────────────────
# 3. PRÉ-TRAITEMENT
# ─────────────────────────────────────────
print("\n=== 3. PRÉ-TRAITEMENT ===")

# --- 3a. Valeurs manquantes ---
print("\n-- Valeurs manquantes --")
nulls = df.isnull().sum()
print(nulls[nulls > 0])

# Supprimer colonnes avec > 50% manquants
seuil = 0.5
cols_a_supprimer = [c for c in df.columns if df[c].isnull().mean() > seuil]
df.drop(columns=cols_a_supprimer, inplace=True)
print(f"Colonnes supprimées (> 50% manquants) : {cols_a_supprimer}")

# Remplacer inf par NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remplir les NaN numériques par la médiane
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

print("Valeurs manquantes après traitement :", df.isnull().sum().sum())

# --- 3b. Doublons ---
print("\n-- Doublons --")
nb_doublons = df.duplicated().sum()
print(f"Doublons détectés : {nb_doublons}")
df.drop_duplicates(inplace=True)
print(f"Lignes après suppression : {len(df):,}")

# --- 3c. Valeurs aberrantes (méthode IQR — boîte à moustache) ---
print("\n-- Valeurs aberrantes (IQR) --")

colonnes_numeriques = df.select_dtypes(include=[np.number]).columns

for col in colonnes_numeriques:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR
    nb_outliers = ((df[col] < borne_basse) | (df[col] > borne_haute)).sum()
    if nb_outliers > 0:
        df[col] = df[col].clip(borne_basse, borne_haute)

print("Outliers traités par clipping (IQR).")

# Visualisation boîte à moustache (10 premières colonnes numériques)
colonnes_viz = list(colonnes_numeriques[:10])
df[colonnes_viz].plot(kind="box", figsize=(14, 5), title="Boîtes à moustache")
plt.tight_layout()
plt.savefig("boxplots.png")
print("Graphique sauvegardé : boxplots.png")

# ─────────────────────────────────────────
# 4. SÉLECTION DES VARIABLES + ENCODAGE
# ─────────────────────────────────────────
print("\n=== 4. SÉLECTION & ENCODAGE ===")

# Variables explicatives (X) et cible (y)
FEATURES = [
    "Fwd Packet Length Max", "Flow IAT Std", "Fwd Packet Length Std",
    "Fwd IAT Total", "Flow Packets/s", "Fwd Packet Length Mean",
    "Flow Bytes/s", "Flow IAT Mean", "Bwd Packet Length Mean",
    "Flow IAT Max", "Bwd Packet Length Std",
]

# Garder seulement celles qui existent dans le dataset
features_ok = [c for c in FEATURES if c in df.columns]
X = df[features_ok].copy()
y = df["Label"].copy()

print(f"Features sélectionnées : {features_ok}")
print(f"Variable cible : Label ({y.nunique()} classes)")

# Encodage Label → numérique (LabelEncoder)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nCorrespondance encodage :")
for code, classe in enumerate(le.classes_):
    print(f"  {code} → {classe}")

# ─────────────────────────────────────────
# 5. NORMALISATION (StandardScaler)
# ─────────────────────────────────────────
print("\n=== 5. NORMALISATION ===")

scaler = StandardScaler()
X_normalise = scaler.fit_transform(X)
X_normalise = pd.DataFrame(X_normalise, columns=X.columns)

print("Moyenne après normalisation :")
print(X_normalise.mean().round(4))
print("\nÉcart-type après normalisation :")
print(X_normalise.std().round(4))

# ─────────────────────────────────────────
# 6. TRAIN / TEST SPLIT (80% / 20%)
# ─────────────────────────────────────────
print("\n=== 6. TRAIN / TEST SPLIT ===")

X_train, X_test, y_train, y_test = train_test_split(
    X_normalise, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Train : {len(X_train):,} lignes ({len(X_train)/len(X_normalise)*100:.0f}%)")
print(f"Test  : {len(X_test):,} lignes  ({len(X_test)/len(X_normalise)*100:.0f}%)")

print("\nDistribution des classes dans le train :")
train_labels = le.inverse_transform(y_train)
for cls, cnt in pd.Series(train_labels).value_counts().items():
    print(f"  {cls} : {cnt:,}")

# Sauvegarde
train_df = X_train.copy()
train_df["Label"] = le.inverse_transform(y_train)
test_df = X_test.copy()
test_df["Label"] = le.inverse_transform(y_test)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\n✅ train.csv et test.csv sauvegardés.")
print("✅ Pipeline terminé.")