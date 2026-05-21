# =========================================
# IMPORTS
# =========================================

from fastapi import (

    APIRouter,
    Header
)

from loguru import logger
import numpy as np
import random
import time

# =========================================
# JWT AUTH
# =========================================

from backend.auth.jwt_handler import (

    verify_token
)

# =========================================
# AI SERVICE
# =========================================

from backend.services.ai_service import (

    predict,
    X
)

# =========================================
# ELASTICSEARCH
# =========================================

from backend.services.elasticsearch_service import (

    index_alert
)

# =========================================
# DATABASE
# =========================================

from sqlalchemy import insert

from backend.database.db import (

    SessionLocal
)

from backend.database.models import (

    alerts
)

# =========================================
# ROUTER
# =========================================

router = APIRouter()

# =========================================
# PREDICT ENDPOINT
# =========================================

@router.get("/predict")

def get_prediction(

    authorization: str = Header(
    default=None,
    alias="Authorization"
)
):

    # =====================================
    # VERIFY JWT TOKEN
    # =====================================

    verify_token(

    authorization.replace(

        "Bearer ",

        ""
    )
)

    # =====================================
    # RANDOM SAMPLE
    # =====================================

    sample = X[

        np.random.randint(
            0,
            len(X)
        )
    ]

    # =====================================
    # AI PREDICTION
    # =====================================

    try:

        logger.info(

    "Running AI prediction..."
)

        result = predict(sample)

        logger.success(

    f"Prediction completed: {result['attack_type']}"
)

    except Exception as e:

        return {

            "error": str(e)
        }

    # =====================================
    # SIMULATED IP
    # =====================================

    ip_address = (

        f"192.168.1."

        f"{random.randint(1,255)}"
    )

    # =====================================
    # TRAFFIC VOLUME
    # =====================================

    traffic_volume = random.randint(

        100,

        5000
    )

    # =====================================
    # VALIDATION
    # =====================================

    if traffic_volume < 0:

        return {

            "error":
            "Invalid traffic volume"
        }

    # =====================================
    # TIMESTAMP
    # =====================================

    timestamp = time.strftime(

        "%Y-%m-%d %H:%M:%S"
    )

    # =====================================
    # ATTACK TYPE
    # =====================================

    if result["anomaly_detected"]:

        attack_type = "Anomaly"

    elif result[
        "random_forest_prediction"
    ] == 1:

        attack_type = "Known Attack"

    else:

        attack_type = "Normal"

    # =====================================
    # RISK LEVEL
    # =====================================

    score = result[
        "iforest_score"
    ]

    if score > 0.70:

        risk_level = "High"

    elif score > 0.50:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    # =====================================
    # DATABASE INSERT
    # =====================================

    db = SessionLocal()

    query = insert(alerts).values(

        timestamp=timestamp,

        ip_address=ip_address,

        traffic_volume=traffic_volume,

        attack_type=attack_type,

        risk_level=risk_level,

        iforest_score=float(score)
    )

    db.execute(query)

    db.commit()

    db.close()

    # =====================================
    # ELASTICSEARCH DOCUMENT
    # =====================================

    alert_document = {

        "timestamp":
            timestamp,

        "ip_address":
            ip_address,

        "traffic_volume":
            traffic_volume,

        "attack_type":
            attack_type,

        "risk_level":
            risk_level,

        "iforest_score":
            float(score)
    }

    # =====================================
    # SEND TO ELASTICSEARCH
    # =====================================

    index_alert(alert_document)

    # =====================================
    # FINAL RESPONSE
    # =====================================

    return {

        "timestamp":
            timestamp,

        "ip_address":
            ip_address,

        "traffic_volume":
            traffic_volume,

        "attack_type":
            attack_type,

        "alert_type":
            "Hybrid AI IDS",

        "risk_level":
            risk_level,

        "random_forest_prediction":
            result[
                "random_forest_prediction"
            ],

        "rf_confidence":
            result[
                "rf_confidence"
            ],

        "kmeans_cluster":
            result[
                "kmeans_cluster"
            ],

        "iforest_score":
            result[
                "iforest_score"
            ],

        "anomaly_detected":
            result[
                "anomaly_detected"
            ]
    }