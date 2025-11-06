# Tên file: database.py
import os
import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, select, func, Float, UniqueConstraint
from werkzeug.security import generate_password_hash, check_password_hash

# --- 1. LOAD ENV VARS & SETUP DB CONNECTION ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env file")

ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(ASYNC_DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


# --- 2. FASTAPI DEPENDENCY ---
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session


# --- 3. DEFINE DB MODELS (TABLES) ---

class User(Base):
    """User table model."""
    __tablename__ = 'web_users'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)

    # Relationship: One User has many Conversations
    conversations: Mapped[List["Conversation"]] = relationship(back_populates='user')

    # --- THÊM MỚI ---
    # Relationship: One User has many ChatMessages
    messages: Mapped[List["ChatMessage"]] = relationship(back_populates='user')

    feedback: Mapped[List["Feedback"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Conversation(Base):
    """Conversation table model."""
    __tablename__ = 'web_conversations'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[int] = mapped_column(ForeignKey('web_users.id'), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New Conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # Relationship back to User
    user: Mapped["User"] = relationship(back_populates='conversations')
    # Relationship: One Conversation has many ChatMessages
    messages: Mapped[List["ChatMessage"]] = relationship(back_populates='conversation', cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat message table model."""
    __tablename__ = 'web_chat_messages'
    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey('web_conversations.id'), nullable=False, index=True)

    # --- THÊM MỚI: DÒNG NÀY LÀ QUAN TRỌNG NHẤT ---
    user_id: Mapped[int] = mapped_column(ForeignKey('web_users.id'), nullable=False, index=True)
    # --- KẾT THÚC THÊM MỚI ---

    sender: Mapped[str] = mapped_column(String(10), nullable=False)  # 'user' or 'bot'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationship back to Conversation
    conversation: Mapped["Conversation"] = relationship(back_populates='messages')

    # --- THÊM MỚI ---
    # Relationship back to User
    user: Mapped["User"] = relationship(back_populates='messages')
    # --- KẾT THÚC THÊM MỚI ---

    # Relationship: One Bot Message might have one Disease Detection
    disease_detection: Mapped[Optional["DiseaseDetection"]] = relationship(back_populates='message', uselist=False,
                                                                           cascade="all, delete-orphan")
    feedback: Mapped[List["Feedback"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan")

class DiseaseDetection(Base):
    """Disease detection result table model."""
    __tablename__ = 'disease_detections'
    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey('web_chat_messages.id'), unique=True, nullable=False)
    plant_type: Mapped[Optional[str]] = mapped_column(String(100))
    disease_name: Mapped[Optional[str]] = mapped_column(String(150), index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float)  # Store as 0.xx float
    detected_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # Relationship back to ChatMessage
    message: Mapped["ChatMessage"] = relationship(back_populates='disease_detection')


class Feedback(Base):
    """Lưu trữ phản hồi của người dùng cho một tin nhắn cụ thể."""
    __tablename__ = "web_feedback"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Khóa ngoại tới tin nhắn được phản hồi
    message_id: Mapped[int] = mapped_column(ForeignKey("web_chat_messages.id"), nullable=False, index=True)

    # Khóa ngoại tới người dùng đã phản hồi
    user_id: Mapped[int] = mapped_column(ForeignKey("web_users.id"), nullable=False, index=True)

    # Xếp hạng: 1 = Tốt (thumbs up), -1 = Tệ (thumbs down)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)

    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Mối quan hệ
    message: Mapped["ChatMessage"] = relationship(back_populates="feedback")
    user: Mapped["User"] = relationship(back_populates="feedback")

    # Đảm bảo mỗi user chỉ có thể feedback 1 LẦN cho 1 message
    __table_args__ = (UniqueConstraint('message_id', 'user_id', name='uq_user_message_feedback'),)