from fastapi import (

    APIRouter,
    HTTPException
)

from sqlalchemy import select

from loguru import logger

from backend.auth.jwt_handler import (

    verify_token
)

from backend.database.db import (

    SessionLocal
)

from backend.database.models import (

    alerts
)

router = APIRouter()

@router.get("/alerts/{token}")

def get_alerts(token: str):

    try:

        # ===============================
        # VERIFY TOKEN
        # ===============================

        verify_token(token)

        # ===============================
        # DATABASE SESSION
        # ===============================

        db = SessionLocal()

        query = select(alerts)

        result = db.execute(query)

        rows = result.mappings().all()

        db.close()

        # ===============================
        # LOG ALERT COUNT
        # ===============================

        logger.info(

            f"{len(rows)} alerts retrieved"
        )

        return rows

    except Exception as e:

        logger.error(

            f"Alerts retrieval failed: {str(e)}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )