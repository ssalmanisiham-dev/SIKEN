from fastapi import (

    APIRouter,
    HTTPException
)

from loguru import logger

from sqlalchemy import select

from backend.database.db import (

    SessionLocal
)

from backend.database.models import (

    users
)

from backend.auth.jwt_handler import (

    create_token
)

router = APIRouter()

@router.post("/login")

def login(

    username: str,

    password: str
):

    db = SessionLocal()

    logger.info(

    f"Checking login for user: {username}"
)

    query = select(users).where(

        users.c.username == username
    )

    result = db.execute(query)

    user = result.fetchone()

    db.close()

    if user is None:

        raise HTTPException(

            status_code=401,

            detail="Invalid Username"
        )

    if user.password != password:

        raise HTTPException(

            status_code=401,

            detail="Invalid Password"
        )

    token = create_token({

        "user": username
    })

    logger.success(

    f"User {username} logged in successfully"
)

    return {

        "access_token": token
    }