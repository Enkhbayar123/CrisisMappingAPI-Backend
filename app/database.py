import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get the password from the Environment Variables (Docker passes these in)
USER = os.getenv("POSTGRES_USER", "crisis_user")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "secure_password_123")
DB_NAME = os.getenv("POSTGRES_DB", "crisis_db")
# IMPORTANT: In Docker, the hostname is the Service Name ("db"), not "localhost"
HOST = "db" 

SQLALCHEMY_DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}/{DB_NAME}"

# 2. Create the Engine (The connection pool)
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 3. Create the Session (The 'handle' for database transactions)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create the Base Class (All models inherit from this)
Base = declarative_base()