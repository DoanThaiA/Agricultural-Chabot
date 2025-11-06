
from sqladmin import ModelView,expose,BaseView
from starlette.requests import Request
from starlette.templating import Jinja2Templates

from database import User, Conversation, ChatMessage, DiseaseDetection,Feedback


templates = Jinja2Templates(directory="templates")

class RAGManagerView(BaseView):
    name = "RAG Manager"
    icon = "fa-solid fa-database"

    @expose("/admin/rag-manager", methods=["GET"])
    async def rag_manager_page(self, request: Request):
        return templates.TemplateResponse(
            "rag_manager.html",
            {
                "request": request,
                # Add your other context variables here
            }
        )
class UserAdmin(ModelView, model=User):
    """Giao diện quản lý Người dùng"""
    column_list = ["id", "username", "email", "created_at"]
    column_searchable_list = ["username", "email"]
    column_sortable_list = ["created_at"]

    #CHỈ cho phép sửa và xóa
    can_create = False
    can_edit = True
    can_delete = True

    # CHỈ dùng form_columns (bỏ form_excluded_columns)
    form_columns = ["username", "email"]

    column_details_list = ["id", "username", "email", "created_at"]

    name = "Người dùng"
    name_plural = "Người dùng"
    icon = "fa-solid fa-user"


class ConversationAdmin(ModelView, model=Conversation):
    column_list = ["id", "user", "title", "created_at"]
    column_searchable_list = ["title"]
    column_sortable_list = ["created_at"]

    name = "Conversation"
    name_plural = "Conversations"
    icon = "fa-solid fa-comments"


class ChatMessageAdmin(ModelView, model=ChatMessage):
    column_list = ["id", "conversation", "sender", "content", "timestamp"]
    column_searchable_list = ["content"]
    column_sortable_list = ["timestamp"]

    # ✅ BỎ column_filters nếu không cần thiết
    # column_filters = ["sender"]

    can_create = False
    name = "Message"
    name_plural = "Chat Messages"
    icon = "fa-solid fa-comment-dots"


class DiseaseDetectionAdmin(ModelView, model=DiseaseDetection):
    column_list = ["id", "message", "plant_type", "disease_name", "confidence", "detected_at"]
    column_searchable_list = ["plant_type", "disease_name"]
    column_sortable_list = ["detected_at", "confidence"]

    # ✅ BỎ column_filters nếu không cần thiết
    # column_filters = ["plant_type", "disease_name"]

    can_create = False
    can_edit = False
    can_delete = False
    name = "Detection"
    name_plural = "Disease Detections"
    icon = "fa-solid fa-leaf"


class FeedbackAdmin(ModelView, model=Feedback):
    name = "Phản hồi"
    name_plural = "Các phản hồi"
    icon = "fa-solid fa-thumbs-up"

    column_list = ["id", "user", "message", "rating", "comment", "created_at"]

    column_searchable_list = ["comment"]

    # ✅ Cột sắp xếp: Dùng chuỗi
    column_sortable_list = ["created_at", "rating"]

    # Chỉ cho phép xem, không cho admin tự tạo/sửa
    can_create = False
    can_edit = False