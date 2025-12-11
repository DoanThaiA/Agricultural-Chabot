import asyncio
import uuid

import uvicorn
import json
import traceback
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
import shutil
import os
from agents.vector_store import process_document_background
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.propagate = False
from sqladmin import Admin
from sqladmin.authentication import AuthenticationBackend
from admin import UserAdmin, ConversationAdmin, ChatMessageAdmin, DiseaseDetectionAdmin, FeedbackAdmin, RAGManagerView
from database import Base, engine, get_db_session, AsyncSession, User, Conversation, ChatMessage, DiseaseDetection,Feedback
from chatbot_service import AgricultureChatbot


# --- 2. CẤU HÌNH ADMIN AUTH ---
class AdminAuth(AuthenticationBackend):
    def __init__(self, secret_key: str):
        super().__init__(secret_key=secret_key)  # thêm dòng này

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        if username == "admin" and password == "12345":
            request.session["admin_user"] = "admin"
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return "admin_user" in request.session

async def get_admin_user(request: Request):
    """
    Dependency (phụ thuộc) của FastAPI để bảo vệ API.
    Nó kiểm tra xem admin đã đăng nhập qua session chưa.
    """
    if not await authentication_backend.authenticate(request):
        # Nếu chưa xác thực, ném lỗi 403 Forbidden
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized"
        )
    # Nếu đã xác thực, trả về tên user admin
    return request.session["admin_user"]


# --- 3. FASTAPI LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: Creating DB tables...")
    async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all)
    logger.info("DB tables OK.")
    os.makedirs("../temp_uploads", exist_ok=True)
    os.makedirs("../temp_images", exist_ok=True)
    yield
    logger.info("Shutdown.")


# --- 4. KHỞI TẠO APP ---
app = FastAPI(
    title="Agriculture Chatbot API",
    version="1.0",
    description="API for the AI-powered Agriculture Chatbot",
    lifespan=lifespan
)
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey123")
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
# --- 5. CẤU HÌNH CORS ---
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- 7. ADMIN PANEL ---
authentication_backend = AdminAuth(secret_key=APP_SECRET_KEY)
admin = Admin(app, engine, authentication_backend=authentication_backend)
admin.add_view(UserAdmin)
admin.add_view(ConversationAdmin)
admin.add_view(ChatMessageAdmin)
admin.add_view(DiseaseDetectionAdmin)
admin.add_view(RAGManagerView)
admin.add_view(FeedbackAdmin)
logger.info("Admin panel mounted at /admin")

# --- 8. Pydantic models ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    user_id: int
    message: str
    conversation_id: Optional[str] = None
    image_data: Optional[str] = Field(None)


class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageInfo(BaseModel):
    id: int
    sender: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True


class DiseaseDetectionInfo(BaseModel):
    id: int
    message_id: int
    plant_type: Optional[str]
    disease_name: Optional[str]
    confidence: Optional[float]
    detected_at: datetime
    conversation_id: str

    class Config:
        from_attributes = True

class DeleteRequest(BaseModel):
    user_id: int
class FeedbackCreate(BaseModel):
    message_id: int
    user_id: int
    rating: int  # 1 cho 'good', -1 cho 'bad'
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    user_id: int
    rating: int
    comment: Optional[str] = None

    class Config:
        from_attributes = True
# --- 9. API ENDPOINTS ---
@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(User).where((User.username == user_in.username) | (User.email == user_in.email)))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Tên đăng nhập hoặc email đã tồn tại")
    new_user = User(username=user_in.username, email=user_in.email)
    new_user.set_password(user_in.password)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


@app.post("/login", response_model=UserResponse)
async def login(user_in: UserLogin, db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(User).where(User.username == user_in.username))
    user = result.scalars().first()
    if not user or not user.check_password(user_in.password):
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
    return user


@app.post("/chat")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db_session)):
    """Handle chatbot interaction with streaming response"""
    chatbot_service = AgricultureChatbot(db)
    user = await db.get(User, request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không thấy người dung")

    try:
        conversation_id = await chatbot_service.get_or_create_conversation(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            title=request.message
        )

        stream_generator = chatbot_service.process_query(
            user_id=request.user_id,
            user_query=request.message,
            conversation_id=conversation_id,
            image_data=request.image_data
        )

        async def response_generator():
            try:
                async for chunk in stream_generator:
                    yield chunk
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

        return StreamingResponse(response_generator(), media_type="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{user_id}", response_model=List[ConversationInfo])
async def get_conversations(user_id: int, db: AsyncSession = Depends(get_db_session)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    result = await db.execute(
        select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


@app.get("/history/{conversation_id}", response_model=List[MessageInfo])
async def get_conversation_history(conversation_id: str, db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(Conversation).filter(Conversation.id == conversation_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại")
    result = await db.execute(
        select(ChatMessage).filter(ChatMessage.conversation_id == conversation_id).order_by(ChatMessage.timestamp.asc())
    )
    return result.scalars().all()


@app.get("/users/{user_id}/detections", response_model=List[DiseaseDetectionInfo])
async def get_user_disease_detections(user_id: int, db: AsyncSession = Depends(get_db_session)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")

    stmt = (
        select(
            DiseaseDetection.id,
            DiseaseDetection.message_id,
            DiseaseDetection.plant_type,
            DiseaseDetection.disease_name,
            DiseaseDetection.confidence,
            DiseaseDetection.detected_at,
            Conversation.id.label("conversation_id"),
        )
        .join(ChatMessage, DiseaseDetection.message_id == ChatMessage.id)
        .join(Conversation, ChatMessage.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
        .order_by(DiseaseDetection.detected_at.desc())
    )
    result = await db.execute(stmt)
    detections = result.mappings().all()
    return detections


@app.delete("/conversations/{conversation_id}", tags=["Chat History"])
async def delete_conversation(
        conversation_id: str,
        request: DeleteRequest,  # Lấy user_id từ body
        db: AsyncSession = Depends(get_db_session)
):
    """Xóa một hội thoại và tất cả tin nhắn bên trong."""

    # 1. Lấy hội thoại từ DB
    convo = await db.get(Conversation, conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. KIỂM TRA BẢO MẬT: Đảm bảo đúng chủ sở hữu
    if convo.user_id != request.user_id:
        raise HTTPException(status_code=403, detail="Access denied: You do not own this conversation")

    # 3. Xóa (database.py đã có 'cascade="all, delete-orphan"')
    await db.delete(convo)
    await db.commit()

    return {"message": "Conversation deleted successfully", "conversation_id": conversation_id}


@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(feedback_in: FeedbackCreate, db: AsyncSession = Depends(get_db_session)):
    """Nhận và lưu trữ phản hồi của người dùng cho một tin nhắn."""

    # 1. Kiểm tra xem message có tồn tại không
    message = await db.get(ChatMessage, feedback_in.message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # 2. Kiểm tra bảo mật: user gửi feedback có phải là chủ của hội thoại không
    conversation = await db.get(Conversation, message.conversation_id)
    if not conversation or conversation.user_id != feedback_in.user_id:
        raise HTTPException(status_code=403, detail="Access denied: You do not own this conversation")

    # 3. Kiểm tra xem user đã feedback tin nhắn này chưa (Update or Create)
    stmt = select(Feedback).where(
        Feedback.message_id == feedback_in.message_id,
        Feedback.user_id == feedback_in.user_id
    )
    result = await db.execute(stmt)
    existing_feedback = result.scalars().first()

    if existing_feedback:
        existing_feedback.rating = feedback_in.rating
        existing_feedback.comment = feedback_in.comment
        db.add(existing_feedback)
        await db.commit()
        await db.refresh(existing_feedback)
        return existing_feedback
    else:
        # Nếu chưa có -> Tạo mới
        new_feedback = Feedback(
            message_id=feedback_in.message_id,
            user_id=feedback_in.user_id,
            rating=feedback_in.rating,
            comment=feedback_in.comment
        )
        db.add(new_feedback)
        await db.commit()
        await db.refresh(new_feedback)
        return new_feedback


@app.post("/api/upload-document", tags=["Admin RAG Management"], status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        admin_user: str = Depends(get_admin_user)
):
    if not file.filename.endswith(('.pdf', '.txt', '.md', '.docx')):  # Hỗ trợ thêm
        raise HTTPException(status_code=400, detail="Loại file không hợp lệ. Chỉ chấp nhận .pdf, .txt, .md, .docx")

    temp_dir = "../temp_uploads"
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Không thể lưu file upload: {e}")
        raise HTTPException(status_code=500, detail="Không thể lưu file lên server.")
    finally:
        file.file.close()

    asyncio.create_task(process_document_background(temp_path, file.filename))
    logger.info(f"Admin {admin_user} đã tải lên file: {file.filename}. Đang xử lý trong nền.")
    return {"message": f"Đã nhận file '{file.filename}'. Quá trình xử lý (embedding) đang chạy trong nền."}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/admin")
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print("Docs: http://127.0.0.1:8000/docs")
    print("Admin: http://127.0.0.1:8000/admin (admin / 12345)")
    uvicorn.run("app:app", host="0.0.0.0", port=8000)