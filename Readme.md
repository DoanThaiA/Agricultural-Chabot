# Chatbot ứng dụng trí tuệ nhân tạo hỗ trợ nhận diện bệnh cây trồng và tư vấn nông nghiệp

Hệ thống Chatbot hỗ trợ nông nghiệp thông minh, tích hợp công nghệ **Computer Vision** (nhận diện bệnh qua ảnh) và **RAG (Retrieval-Augmented Generation)** để tư vấn nông nghiệp

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Infrastructure-Docker-2496ED?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-4169E1?logo=postgresql&logoColor=white)

---

##  Tính năng 

### 1. Đối với người dùng (Nông dân)
* **Chẩn đoán bệnh cây trồng:** Upload ảnh lá cây, hệ thống sử dụng mô hình Deep Learning để phát hiện bệnh và đưa ra độ tin cậy.
* **Tư vấn hỏi đáp (Chatbot):** Hỏi đáp về kiến thức nông nghiệp
* **Lịch sử:** Xem lại các cuộc hội thoại và kết quả chẩn đoán cũ.

### 2. Đối với Quản trị viên (Admin)
* **Admin Dashboard:** Quản lý người dùng, xem toàn bộ lịch sử chat và phản hồi (Feedback).
* **Knowledge Base Management:** Upload tài liệu PDF/Text để cập nhật kiến thức cho AI (RAG).

---

## 📂 Cấu trúc dự án

```text
MY-PROJECT/
├── docker-compose.yml      
├── .env                                
├── README.md               
│
├── backend/              
│   ├── Dockerfile     
│   ├── requirements.txt
│   ├── app.py  
|   ├── chatbot_service.py   
│   ├── database.py          
│   ├── graph.py
│   ├── agents/              
|   ├ |── chroma_db_storage
|   │ ├── predict_image.py
|   │ ├── text_analyzer.py
|   │ └── vector_store.py
│   |
|   ├──model/
│   ├ ├──disease_model.pth 
│   ├── .dockerignore
|   └── load_json.py
│
├── frontend/                
│   ├── Dockerfile        
│   ├── requirements.txt
│   └── streamlit_app.py
│
└── temp_images/     
```   
---

##  Yêu cầu cài đặt

Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:
1.  **Docker Desktop:** [Tải tại đây](https://www.docker.com/products/docker-desktop/) (Bắt buộc).
2.  **Git:** Để quản lý mã nguồn.

---
##  Hướng dẫn cài đặt & Chạy dự án

### Bước 1: Chuẩn bị Model AI
Do file model (`.pth`) có dung lượng lớn (>100MB) nên không được lưu trên Git. Bạn cần tải file model và đặt thủ công vào đúng vị trí:

* **Tên file:** `disease_model.pth`
* **Thư mục đích:** `backend/model/`
* **Kiểm tra:** Đảm bảo đường dẫn là `backend/model/disease_model.pth`.

### Bước 2: Cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc của dự án (ngang hàng với `docker-compose.yml`) và điền nội dung sau:

```ini
# --- Cấu hình Database ---
DB_USER=postgres
DB_PASSWORD=12345
DB_NAME=agriculture_db

# --- Bảo mật ứng dụng ---
APP_SECRET_KEY=

# --- API Keys ---

COHERE_API_KEY=

TAVILY_API_KEY=

# --- Cấu hình Docker---
BACKEND_API_URL=http://backend:8000
EMBED_MODEL=AITeamVN/Vietnamese_Embedding
```
### Bước 3: Khởi chạy hệ thống
Mở Terminal (CMD/PowerShell) tại thư mục gốc dự án và chạy lệnh:

```bash
docker-compose up --build
```
### Hướng dẫn sử dụng

Sau khi khởi động thành công, bạn có thể truy cập các dịch vụ qua trình duyệt:
* **Chatbot (Frontend)**: http://localhost:8501
* **Admin Panel:** http://localhost:8000/admin (admin / 12345)
* **API Docs:** http://localhost:8000/docs