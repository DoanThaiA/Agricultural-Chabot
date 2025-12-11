from sqladmin import ModelView, expose, BaseView
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from database import User, Conversation, ChatMessage, DiseaseDetection, Feedback
from sqlalchemy import func, select
import sqladmin
templates = Jinja2Templates(directory="templates")


class RAGManagerView(BaseView):
    """Giao diện quản lý RAG Knowledge Base"""
    name = "📚 Quản lý RAG"

    @expose("/admin/rag-manager", methods=["GET"])
    async def rag_manager_page(self, request: Request):
        return templates.TemplateResponse(
            "rag_manager.html",
            {"request": request}
        )


class UserAdmin(ModelView, model=User):
    """Quản lý Người dùng"""
    # Tên hiển thị
    name = "👤 Người dùng"
    name_plural = "👥 Người dùng"

    # Cấu hình hiển thị
    column_list = ["id", "username", "email"]
    column_searchable_list = ["username", "email"]
    column_sortable_list = ["id", "username", "email"]

    # Cấu hình form
    can_create = False
    can_edit = True
    can_delete = True
    can_view_details = True

    form_columns = ["username", "email"]
    column_details_list = ["id", "username", "email"]

    # Labels tiếng Việt
    column_labels = {
        "id": "ID",
        "username": "Tên đăng nhập",
        "email": "Email"
    }

    # Format cột
    column_formatters = {
        "created_at": lambda m, a: m.created_at.strftime("%d/%m/%Y %H:%M") if m.created_at else ""
    }

    # Số lượng item mỗi trang
    page_size = 20
    page_size_options = [10, 20, 50, 100]
class ConversationAdmin(ModelView, model=Conversation):
    """Quản lý Hội thoại"""
    name = "💬 Hội thoại"
    name_plural = "💬 Hội thoại"

    column_list = ["id", "user", "title", "created_at"]
    column_searchable_list = ["title"]
    column_sortable_list = ["id", "created_at"]
    column_default_sort = [("created_at", True)]

    can_create = False
    can_edit = False
    can_delete = True
    can_view_details = True

    column_labels = {
        "id": "ID",
        "user": "Người dùng",
        "title": "Tiêu đề",
        "created_at": "Ngày tạo"
    }

    column_formatters = {
        "created_at": lambda m, a: m.created_at.strftime("%d/%m/%Y %H:%M") if m.created_at else "",
        "title": lambda m, a: m.title[:50] + "..." if m.title and len(m.title) > 50 else m.title
    }

    page_size = 25
    page_size_options = [10, 25, 50, 100]


class ChatMessageAdmin(ModelView, model=ChatMessage):
    """Quản lý Tin nhắn Chat"""
    name = "✉️ Tin nhắn"
    name_plural = "✉️ Tin nhắn"

    column_list = ["id", "conversation", "sender", "content", "timestamp"]
    column_searchable_list = ["content"]
    column_sortable_list = ["id", "timestamp"]
    column_default_sort = [("timestamp", True)]

    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True

    column_labels = {
        "id": "ID",
        "conversation": "Hội thoại",
        "sender": "Người gửi",
        "content": "Nội dung",
        "timestamp": "Thời gian"
    }

    column_formatters = {
        "timestamp": lambda m, a: m.timestamp.strftime("%d/%m/%Y %H:%M:%S") if m.timestamp else "",
        "content": lambda m, a: m.content[:100] + "..." if m.content and len(m.content) > 100 else m.content,
        "sender": lambda m, a: "🤖 Bot" if m.sender == "bot" else "👤 User"
    }

    page_size = 30
    page_size_options = [10, 30, 50, 100]


class DiseaseDetectionAdmin(ModelView, model=DiseaseDetection):
    """Quản lý Phát hiện Bệnh"""
    name_plural = "🌿 Phát hiện bệnh"

    column_list = ["id", "message", "disease_name", "confidence", "detected_at"]
    column_searchable_list = ["disease_name"]
    column_sortable_list = ["id", "detected_at", "confidence"]
    column_default_sort = [("detected_at", True)]

    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True

    column_labels = {
        "id": "ID",
        "message": "Tin nhắn",
        "disease_name": "Tên bệnh",
        "confidence": "Độ tin cậy",
        "detected_at": "Thời gian phát hiện"
    }

    column_formatters = {
        "detected_at": lambda m, a: m.detected_at.strftime("%d/%m/%Y %H:%M") if m.detected_at else "",
        "confidence": lambda m, a: f"{m.confidence:.2%}" if m.confidence else "N/A",
        "plant_type": lambda m, a: m.plant_type.title() if m.plant_type else "Không xác định",
        "disease_name": lambda m, a: m.disease_name.title() if m.disease_name else "Không xác định"
    }

    page_size = 20
    page_size_options = [10, 20, 50, 100]


class FeedbackAdmin(ModelView, model=Feedback):
    """Quản lý Phản hồi"""
    name = "⭐ Phản hồi"
    name_plural = "⭐ Phản hồi"

    column_list = ["id", "user", "message", "rating", "comment", "created_at"]
    column_searchable_list = ["comment"]
    column_sortable_list = ["id", "created_at", "rating"]
    column_default_sort = [("created_at", True)]

    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True

    column_labels = {
        "id": "ID",
        "user": "Người dùng",
        "message": "Tin nhắn",
        "rating": "Đánh giá",
        "comment": "Nhận xét",
        "created_at": "Ngày tạo"
    }

    column_formatters = {
        "created_at": lambda m, a: m.created_at.strftime("%d/%m/%Y %H:%M") if m.created_at else "",
        "rating": lambda m, a: "⭐" * m.rating if m.rating else "Chưa đánh giá",
        "comment": lambda m, a: m.comment[:80] + "..." if m.comment and len(m.comment) > 80 else m.comment
    }

    page_size = 25
    page_size_options = [10, 25, 50, 100]
