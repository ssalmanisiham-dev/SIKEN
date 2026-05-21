# =========================================
# IMPORTS
# =========================================

from sqlalchemy import *

from backend.database.db import (

    engine
)

# =========================================
# METADATA
# =========================================

metadata = MetaData()

# =========================================
# ALERTS TABLE
# =========================================

alerts = Table(

    "alerts",

    metadata,

    Column(
        "id",
        Integer,
        primary_key=True
    ),

    Column(
        "timestamp",
        String
    ),

    Column(
        "ip_address",
        String
    ),

    Column(
        "traffic_volume",
        Integer
    ),

    Column(
        "attack_type",
        String
    ),

    Column(
        "risk_level",
        String
    ),

    Column(
        "iforest_score",
        Float
    )
)

# =========================================
# USERS TABLE
# =========================================

users = Table(

    "users",

    metadata,

    Column(
        "id",
        Integer,
        primary_key=True
    ),

    Column(
        "username",
        String,
        unique=True
    ),

    Column(
        "password",
        String
    )
)

# =========================================
# CREATE TABLES
# =========================================

metadata.create_all(engine)