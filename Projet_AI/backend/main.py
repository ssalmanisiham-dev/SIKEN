# =========================================
# FASTAPI IMPORT
# =========================================

from fastapi import FastAPI

from sqlalchemy import insert

from loguru import logger

from backend.database.db import (

    SessionLocal
)

from backend.database.models import (

    users
)

# =========================================
# AUTH ROUTE
# =========================================

from backend.routes.auth import (

    router as auth_router
)

# =========================================
# ALERTS ROUTE
# =========================================

from backend.routes.alerts import (

    router as alerts_router
)

# =========================================
# PREDICT ROUTE
# =========================================

from backend.routes.predict import (

    router
)

# =========================================
# DATABASE
# =========================================

from backend.database import models

# =========================================
# FASTAPI APP
# =========================================

app = FastAPI(

    title="Hybrid AI IDS",

    version="1.0"
)

logger.info(

    "Hybrid AI IDS Started Successfully"
)
# =========================================
# ROUTES
# =========================================

app.include_router(router)

app.include_router(auth_router)

app.include_router(alerts_router)

# =========================================
# HOME ENDPOINT
# =========================================

@app.get("/")

def home():

    return {

        "message":
            "Hybrid AI IDS Running",

        "status":
            "online"
    }

db = SessionLocal()

query = insert(users).values(

    username="admin",

    password="admin123"
)

try:

    db.execute(query)

    db.commit()

except:

    pass

db.close()