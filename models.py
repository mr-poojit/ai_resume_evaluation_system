from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

# Load DB URL from environment variable or use default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Chat query history table
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

# Manual problem table
class ManualProblem(Base):
    __tablename__ = "manual_problems"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)

# Create tables
Base.metadata.create_all(bind=engine)
