# =========================================
# SQLALCHEMY IMPORTS
# =========================================

from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker

# =========================================
# DATABASE URL
# =========================================

DATABASE_URL = (

    "sqlite:///./alerts.db"
)

# =========================================
# CREATE ENGINE
# =========================================

engine = create_engine(

    DATABASE_URL,

    connect_args={

        "check_same_thread": False
    }
)

# =========================================
# SESSION FACTORY
# =========================================

SessionLocal = sessionmaker(

    autocommit=False,

    autoflush=False,

    bind=engine
)