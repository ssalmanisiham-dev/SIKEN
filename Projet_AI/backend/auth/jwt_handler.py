# =========================================
# JWT IMPORTS
# =========================================

from jose import jwt

from datetime import (

    datetime,
    timedelta
)

from fastapi import (

    HTTPException
)

# =========================================
# SECRET CONFIG
# =========================================

SECRET_KEY = "hybrid_ai_ids"

ALGORITHM = "HS256"

# =========================================
# CREATE TOKEN
# =========================================

def create_token(data):

    payload = data.copy()

    payload["exp"] = (

        datetime.utcnow()

        + timedelta(hours=2)
    )

    token = jwt.encode(

        payload,

        SECRET_KEY,

        algorithm=ALGORITHM
    )

    return token

# =========================================
# VERIFY TOKEN
# =========================================

def verify_token(token):

    try:

        payload = jwt.decode(

            token,

            SECRET_KEY,

            algorithms=[ALGORITHM]
        )

        return payload

    except Exception:

        raise HTTPException(

            status_code=401,

            detail="Invalid Token"
        )