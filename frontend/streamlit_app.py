import os

import streamlit as st
import requests
import base64
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

# =============================================================================
# CẤU HÌNH ỨNG DỤNG
# =============================================================================

st.set_page_config(
    page_title="Trợ lý Nông nghiệp AI 🌱",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
API_ENDPOINTS = {
    "login": f"{API_BASE_URL}/login",
    "register": f"{API_BASE_URL}/register",
    "chat": f"{API_BASE_URL}/chat",
    "conversations": f"{API_BASE_URL}/conversations",
    "history": f"{API_BASE_URL}/history",
    "disease": f"{API_BASE_URL}/users",
    "feedback": f"{API_BASE_URL}/feedback"
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
body {
    background-color: #f7f9fb;
    font-family: "Inter", sans-serif;
}

/* Form đăng nhập & đăng ký */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 90vh;
}

.form-box {
    background-color: white;
    border: 1px solid #e3e6ea;
    border-radius: 12px;
    padding: 2rem 3rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    width: 400px;
    text-align: center;
}

.form-box h2 {
    color: #2b3e50;
    margin-bottom: 1rem;
}

.stTextInput>div>div>input {
    background-color: #f9fafb;
    border: 1px solid #d6dee6;
    border-radius: 8px;
    padding: 0.6rem;
}

.stTextInput>div>div>input:focus {
    border-color: #10a37f;
    box-shadow: 0 0 0 2px rgba(16,163,127,0.3);
}

.stButton>button {
    background-color: #10a37f;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    width: 100%;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    background-color: #0d8b6f;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e3e6ea;
    padding: 1.5rem;
}

[data-testid="stSidebar"] h1 {
    color: #2b3e50;
    text-align: center;
}

.sidebar-btn {
    width: 100%;
    background-color: #f2f4f6;
    color: #2b3e50;
    border: 1px solid #e3e6ea;
    border-radius: 10px;
    padding: 0.6rem;
    margin: 0.4rem 0;
    text-align: center;
    font-weight: 500;
    transition: all 0.2s ease;
}

.sidebar-btn:hover {
    background-color: #10a37f;
    color: white;
}

/* Chat */
.stChatMessage {
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
}

.stChatMessage[data-testid="stChatMessageUser"] {
    background-color: #e6f4ee;
    color: #1c4532;
    margin-left: auto;
}

.stChatMessage[data-testid="stChatMessageAssistant"] {
    background-color: #f9fafb;
    border: 1px solid #e3e6ea;
    color: #2b3e50;
    margin-right: auto;
}

/* Chat input */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 1.5rem;
    left: 18rem;
    right: 2rem;
}

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    border-radius: 10px;
    border: 1px solid #d6dee6;
    color: #2b3e50;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Khởi tạo tất cả session state variables"""
    defaults = {
        "user_id": None,
        "username": None,
        "messages": [],
        "conversation_id": None,
        "conversation_list": [],
        "view_mode": "chat",
        "disease_history": [],
        "show_success_message": False,
        "success_username": None,
        "uploaded_file": None,
        "message_images": {}  # Dictionary để lưu ảnh theo message index
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# API FUNCTIONS
# =============================================================================

def api_request(endpoint: str, method: str = "GET", json_data: Optional[Dict] = None) -> Optional[requests.Response]:
    """Hàm helper để gọi API với error handling"""
    try:
        if method == "GET":
            return requests.get(endpoint)
        elif method == "POST":
            return requests.post(endpoint, json=json_data)
        elif method == "DELETE":
            return requests.delete(endpoint, json=json_data)
    except requests.exceptions.ConnectionError:
        st.error("🔌 Không thể kết nối đến server. Vui lòng kiểm tra kết nối!")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        return None


def handle_feedback(message_id: int, rating: int, comment: str = ""):
    """Gửi feedback cho tin nhắn"""
    if not st.session_state.user_id:
        return

    response = api_request(
        API_ENDPOINTS["feedback"],
        "POST",
        {
            "message_id": message_id,
            "user_id": st.session_state.user_id,
            "rating": rating,
            "comment": comment
        }
    )

    if response and response.status_code == 201:
        st.toast("Cảm ơn phản hồi của bạn!", icon="✅")
    else:
        st.toast("❌ Không thể gửi phản hồi", icon="❌")


def handle_login(username: str, password: str) -> bool:
    if not username or not password:
        st.sidebar.error("⚠️ Vui lòng nhập đầy đủ thông tin")
        return False

    with st.spinner("🔐 Đang đăng nhập..."):
        response = api_request(API_ENDPOINTS["login"], "POST", {"username": username, "password": password})

    if response is None:
        st.sidebar.error("❌ Không thể kết nối đến server")
        return False

    try:
        resp_json = response.json()
    except Exception:
        resp_json = {}

    if response.status_code == 200:
        user_data = resp_json
        st.session_state.user_id = user_data["id"]
        st.session_state.username = user_data["username"]
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.view_mode = "chat"
        st.session_state.message_images = {}  # Reset images
        load_conversations()
        st.balloons()
        st.toast(f"🎉 Chào mừng {user_data['username']}!", icon="✅")
        return True
    else:
        error_msg = resp_json.get("detail", f"Lỗi: {response.status_code}")
        st.sidebar.error(f"❌ {error_msg}")
        return False


def handle_register(username: str, email: str, password: str) -> bool:
    """Xử lý đăng ký tài khoản"""
    if not username or not email or not password:
        st.sidebar.error("⚠️ Vui lòng nhập đầy đủ thông tin")
        return False

    if len(password) < 6:
        st.sidebar.error("⚠️ Mật khẩu phải có ít nhất 6 ký tự")
        return False

    if '@' not in email:
        st.sidebar.error("⚠️ Email không hợp lệ")
        return False

    with st.spinner("📝 Đang xử lý đăng ký..."):
        response = api_request(
            API_ENDPOINTS["register"],
            "POST",
            {"username": username, "email": email, "password": password}
        )
    if response is None:
        st.sidebar.error("❌ Không thể kết nối đến server")
        return False

    try:
        resp_json = response.json()
    except Exception:
        resp_json = {}

    if response and response.status_code == 201:
        st.session_state.show_success_message = True
        st.session_state.success_username = username
        return True
    else:
        error_msg = resp_json.get("detail", f"Lỗi: {response.status_code}")
        st.sidebar.error(f"❌ {error_msg}")
        return False


def handle_logout():
    """Xử lý đăng xuất"""
    username = st.session_state.username
    for key in ["user_id", "username", "messages", "conversation_id", "conversation_list", "disease_history",
                "message_images"]:
        st.session_state[key] = [] if key in ["messages", "conversation_list", "disease_history"] else (
            {} if key == "message_images" else None)
    st.session_state.view_mode = "chat"
    st.toast(f"👋 Tạm biệt {username}!", icon="👋")


def load_conversations():
    """Tải danh sách hội thoại"""
    if not st.session_state.user_id:
        return

    response = api_request(f"{API_ENDPOINTS['conversations']}/{st.session_state.user_id}")
    if response and response.status_code == 200:
        st.session_state.conversation_list = response.json()


def load_history(convo_id: int):
    """Tải lịch sử chat"""
    with st.spinner("📜 Đang tải lịch sử..."):
        response = api_request(f"{API_ENDPOINTS['history']}/{convo_id}")

    if response and response.status_code == 200:
        messages = response.json()
        st.session_state.messages = [
            {"role": msg["sender"], "content": msg["content"], "id": msg["id"]}
            for msg in messages
        ]
        st.session_state.conversation_id = convo_id
        st.session_state.view_mode = "chat"
        # Lưu ý: Ảnh sẽ bị mất khi load lại vì chỉ lưu trong session


def load_disease_history():
    """Tải lịch sử phát hiện bệnh"""
    if not st.session_state.user_id:
        return

    with st.spinner("🌿 Đang tải lịch sử bệnh..."):
        response = api_request(f"{API_ENDPOINTS['disease']}/{st.session_state.user_id}/detections")

    if response and response.status_code == 200:
        st.session_state.disease_history = response.json()


def delete_conversation(convo_id: int):
    """Xóa hội thoại"""
    response = api_request(
        f"{API_BASE_URL}/conversations/{convo_id}",
        "DELETE",
        {"user_id": st.session_state.user_id}
    )

    if response and response.status_code == 200:
        st.toast("🗑️ Đã xóa hội thoại!", icon="✅")
        load_conversations()
        if st.session_state.conversation_id == convo_id:
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.message_images = {}  # Xóa ảnh
        return True
    return False


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_auth_sidebar():
    """Render sidebar khi chưa đăng nhập"""
    st.sidebar.markdown("# 🌱 Trợ lý Nông nghiệp")
    st.sidebar.markdown("### Hệ thống AI chẩn đoán bệnh cây trồng")
    st.sidebar.markdown("---")

    # Hiển thị thông báo đăng ký thành công
    if st.session_state.show_success_message:
        st.sidebar.success(f"✅ Đăng ký thành công!\n\nChào mừng **{st.session_state.success_username}**!")
        st.sidebar.info("👉 Vui lòng đăng nhập để tiếp tục")
        st.balloons()
        st.session_state.show_success_message = False

    tab1, tab2 = st.sidebar.tabs(["🔐 Đăng nhập", "📝 Đăng ký"])

    with tab1:
        st.markdown("#### Đăng nhập tài khoản")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Tên đăng nhập", placeholder="Nhập tên đăng nhập")
            password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Nhập mật khẩu")
            login_btn = st.form_submit_button("Đăng nhập", use_container_width=True, type="primary")

            if login_btn:
                if handle_login(username, password):
                    st.rerun()

    with tab2:
        st.markdown("#### Tạo tài khoản mới")
        with st.form("register_form", clear_on_submit=False):
            reg_username = st.text_input("👤 Tên đăng nhập", placeholder="Chọn tên đăng nhập")
            reg_email = st.text_input("📧 Email", placeholder="email@example.com")
            reg_password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Ít nhất 6 ký tự")
            st.caption("💡 Mật khẩu nên có ít nhất 6 ký tự, bao gồm chữ và số")
            register_btn = st.form_submit_button("Đăng ký", use_container_width=True, type="primary")

            if register_btn:
                if handle_register(reg_username, reg_email, reg_password):
                    st.rerun()


def render_user_sidebar():
    """Render sidebar khi đã đăng nhập"""
    # User Info
    st.sidebar.markdown(f"### 👋 Xin chào!")
    st.sidebar.markdown(f"**{st.session_state.username}**")

    if st.sidebar.button("🚪 Đăng xuất", use_container_width=True, type="secondary"):
        handle_logout()
        st.rerun()

    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.markdown("### 📍 Điều hướng")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button(
                "💬 Chat",
                use_container_width=True,
                type="primary" if st.session_state.view_mode == "chat" else "secondary",
                help="Trò chuyện với AI"
        ):
            st.session_state.view_mode = "chat"
            st.rerun()

    with col2:
        if st.button(
                "🌿 Lịch sử",
                use_container_width=True,
                type="primary" if st.session_state.view_mode == "disease" else "secondary",
                help="Xem lịch sử phát hiện bệnh"
        ):
            st.session_state.view_mode = "disease"
            load_disease_history()
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("➕ Đoạn chat mới", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.message_images = {}  # Xóa ảnh
        st.rerun()
    st.sidebar.markdown("### 💭 Lịch sử hội thoại")
    if not st.session_state.conversation_list:
        st.sidebar.info("💡 Chưa có hội thoại nào")
    else:
        st.sidebar.markdown(f"**📊 Tổng: {len(st.session_state.conversation_list)} hội thoại**")

        # Conversation list with scrollable container
        with st.sidebar.container():
            for convo in st.session_state.conversation_list:
                is_active = convo["id"]    == st.session_state.conversation_id

                col1, col2 = st.columns([0.85, 0.15])

                with col1:
                    btn_label = f"{'📌' if is_active else '💬'} {convo['title'][:22]}{'...' if len(convo['title']) > 22 else ''}"

                    if st.button(
                            btn_label,
                            key=f"conv_{convo['id']}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary",
                            help=convo['title']
                    ):
                        load_history(convo["id"])
                        st.rerun()

                with col2:
                    if st.button(
                            "🗑️",
                            key=f"del_{convo['id']}",
                            use_container_width=True,
                            help="Xóa hội thoại"
                    ):
                        if delete_conversation(convo["id"]):
                            st.rerun()


def render_welcome_page():
    """Render trang chào mừng"""
    st.markdown("""
        <div class="welcome-card">
            <h1>🌱 Trợ lý Nông nghiệp AI</h1>
            <p>Hệ thống AI thông minh giúp chẩn đoán bệnh cây trồng</p>
            <p style="margin-top: 2rem; font-size: 1rem;">
                Vui lòng <strong>đăng nhập</strong> hoặc <strong>đăng ký</strong> để bắt đầu sử dụng
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px;">
                <h2>🤖</h2>
                <h4>AI Thông minh</h4>
                <p>Công nghệ AI tiên tiến để chẩn đoán bệnh chính xác</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px;">
                <h2>📸</h2>
                <h4>Phân tích ảnh</h4>
                <p>Chỉ cần chụp ảnh, AI sẽ phân tích ngay lập tức</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px;">
                <h2>📊</h2>
                <h4>Theo dõi lịch sử</h4>
                <p>Lưu trữ và quản lý lịch sử chẩn đoán của bạn</p>
            </div>
        """, unsafe_allow_html=True)


def render_chat_view():
    """Render giao diện chat"""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 💬 Trò chuyện với AI")
        if st.session_state.conversation_id:
            current_convo = next(
                (c for c in st.session_state.conversation_list if c["id"] == st.session_state.conversation_id),
                None
            )
            if current_convo:
                st.caption(f"📝 {current_convo['title']}")
        else:
            st.caption("✨ Hội thoại mới")

    with col2:
        if st.session_state.messages:
            if st.button("🗑️ Xóa chat", use_container_width=True, help="Xóa toàn bộ tin nhắn hiện tại"):
                st.session_state.messages = []
                st.session_state.conversation_id = None
                st.session_state.message_images = {}  # Xóa ảnh
                st.rerun()

    st.markdown("---")

    # Chat History
    if not st.session_state.messages:
        st.info(
            "👋 Xin chào! Tôi là trợ lý AI chuyên về nông nghiệp. Hãy hỏi tôi về bệnh cây trồng hoặc tải ảnh lên để phân tích nhé!")

    for idx, msg in enumerate(st.session_state.messages):
        role = "assistant" if msg["role"] == "bot" else msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # Hiển thị ảnh nếu có
            if idx in st.session_state.message_images:
                st.image(st.session_state.message_images[idx], caption="📸 Ảnh đã gửi", width=300)

            # Feedback buttons for bot messages
            if msg["role"] == "bot" and "id" in msg:
                message_id = msg["id"]
                col1, col2, col3 = st.columns([0.5, 0.5, 10])

                with col1:
                    if st.button("👍", key=f"up_{message_id}_{idx}", help="Hữu ích"):
                        handle_feedback(message_id, 1, "")

                with col2:
                    with st.popover("👎", help="Chưa tốt", use_container_width=True):
                        st.markdown("**Góp ý của bạn**")
                        comment = st.text_area(
                            "Hãy cho chúng tôi biết cách cải thiện",
                            key=f"comment_{message_id}_{idx}",
                            placeholder="Phản hồi của bạn rất quan trọng...",
                            height=100
                        )
                        if st.button("📤 Gửi", key=f"submit_{message_id}_{idx}", type="primary",
                                     use_container_width=True):
                            handle_feedback(message_id, -1, comment)

    # Image Upload Section
    st.markdown("---")
    st.markdown("### 📸 Tải ảnh cây trồng (Tùy chọn)")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Chọn ảnh để AI phân tích bệnh cây trồng",
            type=["jpg", "jpeg", "png"],
            help="Hỗ trợ: JPG, JPEG, PNG (tối đa 10MB)",
            label_visibility="collapsed",
            key=f"file_uploader_{len(st.session_state.messages)}"  # Key thay đổi sau mỗi lần gửi
        )

    if uploaded_file:
        with col2:
            st.image(uploaded_file, caption="✅ Sẵn sàng gửi", use_column_width=True)

    # Chat Input
    if prompt := st.chat_input("💭 Nhập câu hỏi của bạn..."):
        # Add user message
        current_msg_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

            # Hiển thị ảnh nếu có
            if uploaded_file:
                st.image(uploaded_file, caption="📸 Ảnh đã gửi", width=300)
                # Lưu ảnh vào session state
                st.session_state.message_images[current_msg_idx] = uploaded_file.getvalue()

        # Process image if uploaded
        image_data_b64 = None
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_data_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Send to API
        payload = {
            "user_id": st.session_state.user_id,
            "message": prompt,
            "conversation_id": st.session_state.conversation_id,
            "image_data": image_data_b64
        }

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Đang phân tích... ▌")
            full_response = ""

            try:
                response = requests.post(API_ENDPOINTS["chat"], json=payload, stream=True)
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data:"):
                        try:
                            data_str = line[len("data:"):].strip()
                            if not data_str:
                                continue

                            data_json = json.loads(data_str)

                            if data_json.get("event") == "end":
                                full_response = data_json.get("final_message", "❌ Không nhận được phản hồi")
                                message_placeholder.markdown(full_response)
                                st.session_state.conversation_id = data_json.get("conversation_id")
                                load_conversations()

                                if st.session_state.conversation_id:
                                    load_history(st.session_state.conversation_id)
                                    st.rerun()
                                break

                            if data_json.get("event") == "error":
                                full_response = f"❌ LỖI: {data_json.get('detail', 'Lỗi không xác định')}"
                                message_placeholder.markdown(full_response)
                                st.session_state.messages.append({"role": "bot", "content": full_response})
                                break

                        except json.JSONDecodeError:
                            continue

            except requests.exceptions.RequestException as e:
                full_response = f"❌ LỖI KẾT NỐI: {e}"
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "bot", "content": full_response})

def render_disease_history_view():
    """Render giao diện lịch sử bệnh"""
    st.markdown("## 🌿 Lịch sử Phát hiện Bệnh")
    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🔄 Làm mới", use_container_width=True):
            load_disease_history()
            st.rerun()

    with col2:
        if st.button("💬 Về Chat", use_container_width=True):
            st.session_state.view_mode = "chat"
            st.rerun()

    # Load data if not loaded
    if not st.session_state.disease_history:
        load_disease_history()

    if not st.session_state.disease_history:
        st.info("💡 Chưa có lịch sử phát hiện bệnh nào.\n\nHãy bắt đầu chat với AI và tải ảnh cây trồng để phân tích!")

        if st.button("🚀 Bắt đầu chat ngay", type="primary"):
            st.session_state.view_mode = "chat"
            st.rerun()
    else:
        # Statistics
        total_diseases = len(st.session_state.disease_history)
        unique_diseases = len(set(item["disease_name"] for item in st.session_state.disease_history))
        latest_date = datetime.fromisoformat(
            st.session_state.disease_history[0]["detected_at"]
        ).strftime('%d/%m/%Y')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="📊 Tổng phát hiện",
                value=total_diseases,
                help="Tổng số lần phát hiện bệnh"
            )

        with col2:
            st.metric(
                label="🦠 Số loại bệnh",
                value=unique_diseases,
                help="Số loại bệnh khác nhau đã phát hiện"
            )

        with col3:
            st.metric(
                label="📅 Lần cuối",
                value=latest_date,
                help="Ngày phát hiện gần nhất"
            )

        st.markdown("---")

        # Filter options
        with st.expander("🔍 Bộ lọc", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                all_diseases = sorted(set(item["disease_name"] for item in st.session_state.disease_history))
                selected_disease = st.selectbox(
                    "Lọc theo bệnh",
                    ["Tất cả"] + all_diseases,
                    help="Chọn loại bệnh cụ thể"
                )

            with col2:
                all_plants = sorted(
                    set(item["plant_type"] for item in st.session_state.disease_history if item.get("plant_type")))
                selected_plant = st.selectbox(
                    "Lọc theo cây trồng",
                    ["Tất cả"] + all_plants,
                    help="Chọn loại cây trồng"
                )

        # Filter data
        filtered_data = st.session_state.disease_history

        if selected_disease != "Tất cả":
            filtered_data = [item for item in filtered_data if item["disease_name"] == selected_disease]

        if selected_plant != "Tất cả":
            filtered_data = [item for item in filtered_data if item.get("plant_type") == selected_plant]

        # Prepare DataFrame
        data_to_display = []
        for item in filtered_data:
            data_to_display.append({
                "id": item["id"],
                "Ngày": datetime.fromisoformat(item["detected_at"]).strftime('%d-%m-%Y %H:%M'),
                "Tên Bệnh": item["disease_name"],
                "Độ tin cậy": item["confidence"] * 100 if item.get("confidence") else None,
                "conversation_id": item["conversation_id"]
            })

        if not data_to_display:
            st.warning("⚠️ Không tìm thấy kết quả phù hợp với bộ lọc")
        else:
            st.markdown(f"**Hiển thị {len(data_to_display)} kết quả**")

            df = pd.DataFrame(data_to_display)

            # Column configuration
            column_config = {
                "id": None,
                "conversation_id": None,
                "Độ tin cậy": st.column_config.ProgressColumn(
                    "Độ tin cậy (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Ngày": st.column_config.TextColumn(
                    "📅 Ngày phát hiện",
                    width="medium"
                ),
                "Tên Bệnh": st.column_config.TextColumn(
                    "🦠 Tên Bệnh",
                    width="large"
                )
            }

            # Display table
            st.dataframe(
                df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                key="disease_table_selection",
                on_select="rerun",
                selection_mode="single-row",
                height=450
            )

            # Handle selection
            selection_state = st.session_state.get("disease_table_selection")

            if selection_state and selection_state.selection.get("rows"):
                selected_index = selection_state.selection["rows"][0]
                selected_convo_id = df.iloc[selected_index]["conversation_id"]
                selected_disease = df.iloc[selected_index]["Tên Bệnh"]

                st.success(f"✅ Đang tải hội thoại: **{selected_disease}**")
                load_history(selected_convo_id)
                st.session_state.disease_table_selection.selection["rows"] = []
                st.rerun()

            # Export option
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 3])

            with col1:
                csv = df.drop(columns=["id", "conversation_id"]).to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Tải CSV",
                    data=csv,
                    file_name=f"lich_su_benh_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Export to Excel would require openpyxl, keeping CSV only for simplicity
                st.button(
                    "📊 Xem biểu đồ",
                    use_container_width=True,
                    disabled=True,
                    help="Tính năng sắp ra mắt"
                )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application logic"""

    # Sidebar
    if st.session_state.user_id is None:
        render_auth_sidebar()
    else:
        render_user_sidebar()

    # Main content
    if st.session_state.user_id is None:
        render_welcome_page()
    elif st.session_state.view_mode == "chat":
        render_chat_view()
    elif st.session_state.view_mode == "disease":
        render_disease_history_view()


# Run the application
if __name__ == "__main__":
    main()