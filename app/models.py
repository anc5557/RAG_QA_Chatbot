from sqlalchemy import UUID, Column, Integer, String, DateTime, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey
from .database import Base
from datetime import datetime
import enum
import uuid


# Enum for message sender
class SenderEnum(enum.Enum):
    user = "Human"
    bot = "AI"


# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    conversations = relationship("Conversation", back_populates="user")


# Conversation model
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


# Message model
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sender = Column(Enum(SenderEnum), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    conversation = relationship("Conversation", back_populates="messages")
