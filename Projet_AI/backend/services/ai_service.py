# =========================================
# IMPORTS
# =========================================

import pandas as pd
import numpy as np
import joblib
import os

# =========================================
# RANDOM FOREST
# =========================================

from Random_Forest.forest import Forest

# =========================================
# KMEANS
# =========================================

from Kmeans.Kmeans import KMeans

# =========================================
# ISOLATION FOREST
# =========================================

from Isolation_Forest.iforest import (
    IsolationTreeEnsemble
)

# =========================================
# LOAD DATASET
# =========================================

print("\nLoading dataset...")

df = pd.read_csv(

    "data/test.csv",

    nrows=1000
)

# =========================================
# CREATE LABELS
# =========================================

print("\nPreparing labels...")

y = df["Label"].apply(

    lambda x: 0 if x == "BENIGN" else 1
)

# =========================================
# FEATURES
# =========================================

print("\nPreparing features...")

X = df.drop(

    "Label",

    axis=1
)

# =========================================
# CONVERT TO NUMPY
# =========================================

X = X.values

y = y.values

# reshape labels

y = np.atleast_2d(y).T

# =========================================
# RANDOM FOREST MODEL PATH
# =========================================

rf_model_path = "models/rf.pkl"

# =========================================
# LOAD MODEL IF EXISTS
# =========================================

if os.path.exists(rf_model_path):

    print("\nLoading Random Forest model...")

    rf = joblib.load(

        rf_model_path
    )

    print("Random Forest loaded!")

# =========================================
# TRAIN MODEL IF NOT FOUND
# =========================================

else:

    print("\nTraining Random Forest...")

    rf = Forest(

        data=X,

        labels=y,

        n_trees=3,

        max_features=5,

        bootstrap_features=True,

        max_depth=3,

        min_leaf_points=2
    )

    # =====================================
    # SAVE MODEL
    # =====================================

    joblib.dump(

        rf,

        rf_model_path
    )

    print("Random Forest saved!")

# =========================================
# KMEANS MODEL PATH
# =========================================

kmeans_model_path = "models/kmeans.pkl"

# =========================================
# LOAD MODEL IF EXISTS
# =========================================

if os.path.exists(kmeans_model_path):

    print("\nLoading KMeans model...")

    kmeans = joblib.load(

        kmeans_model_path
    )

    print("KMeans loaded!")

# =========================================
# TRAIN MODEL IF NOT FOUND
# =========================================

else:

    print("\nTraining KMeans...")

    kmeans = KMeans(

        k=2
    )

    kmeans.fit(X)

    # =====================================
    # SAVE MODEL
    # =====================================

    joblib.dump(

        kmeans,

        kmeans_model_path
    )

    print("KMeans saved!")

# =========================================
# ISOLATION FOREST MODEL PATH
# =========================================

iforest_model_path = "models/iforest.pkl"

# =========================================
# LOAD MODEL IF EXISTS
# =========================================

if os.path.exists(iforest_model_path):

    print("\nLoading Isolation Forest model...")

    iforest = joblib.load(

        iforest_model_path
    )

    print("Isolation Forest loaded!")

# =========================================
# ISOLATION FOREST MODEL PATH
# =========================================

iforest_model_path = "models/iforest.pkl"

# =========================================
# LOAD MODEL IF EXISTS
# =========================================

if os.path.exists(iforest_model_path):

    print("\nLoading Isolation Forest model...")

    iforest = joblib.load(

        iforest_model_path
    )

    print("Isolation Forest loaded!")

# =========================================
# TRAIN MODEL IF NOT FOUND
# =========================================

else:

    print("\nTraining Isolation Forest...")

    iforest = IsolationTreeEnsemble(

        sample_size=64,

        n_trees=25
    )

    iforest.fit(X)

    # =====================================
    # SAVE MODEL
    # =====================================

    joblib.dump(

        iforest,

        iforest_model_path
    )

    print("Isolation Forest saved!")

# =========================================
# AI PREDICTION FUNCTION
# =========================================

def predict(sample):

    # =====================================
    # RANDOM FOREST
    # =====================================

    rf_pred, rf_prob = rf.predict(

        sample
    )

    # =====================================
    # KMEANS
    # =====================================

    km_cluster = kmeans.predict(

        [sample]
    )[0]

    # =====================================
    # ISOLATION FOREST
    # =====================================

    score = iforest.anomaly_score(

        np.array([sample])
    )[0]

    # =====================================
    # ANOMALY DETECTION
    # =====================================

    anomaly = score > 0.50

    # =====================================
    # ATTACK TYPE
    # =====================================

    if anomaly:

        attack_type = "Anomaly"

    elif rf_pred == 1:

        attack_type = "Known Attack"

    else:

        attack_type = "Normal"

    # =====================================
    # RISK LEVEL
    # =====================================

    if score > 0.70:

        risk_level = "High"

    elif score > 0.50:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    # =====================================
    # RETURN RESULTS
    # =====================================

    return {

        "attack_type":
            attack_type,

        "risk_level":
            risk_level,

        "random_forest_prediction":
            int(rf_pred),

        "rf_confidence":
            float(rf_prob),

        "kmeans_cluster":
            int(km_cluster),

        "iforest_score":
            float(score),

        "anomaly_detected":
            bool(anomaly)
    }

# =========================================
# TEST AI SERVICE
# =========================================

print("\nTesting AI Service...")

sample = X[0]

result = predict(sample)

print("\nPrediction Result :")

print(result)