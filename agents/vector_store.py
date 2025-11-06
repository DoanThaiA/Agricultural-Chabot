# Tên file: vector_store_utils.py

import os
import logging
import traceback
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from fastapi.concurrency import run_in_threadpool

# --- 0. CẤU HÌNH LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- 1. TẢI CẤU HÌNH TỪ .env ---
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    r"C:\Laptrinhweb\32_Thai\pythonProject\agents\chroma_db_storage"
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "AITeamVN/Vietnamese_Embedding")

if not CHROMA_DB_PATH or not EMBED_MODEL:
    raise ValueError("CHROMA_DB_PATH hoặc EMBED_MODEL chưa được thiết lập trong .env")

# --- 2. KHỞI TẠO EMBEDDING & VECTOR STORE ---
try:
    logger.info(f"🚀 Đang khởi tạo mô hình embedding: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # giúp tính cosine chính xác
    )
    logger.info("✅ Embedding model đã sẵn sàng.")

    logger.info(f"📦 Đang load/tạo vector store tại: {CHROMA_DB_PATH}")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"✅ Vector store OK. Tổng số vector hiện có: {vector_store._collection.count()}")

except Exception as e:
    logger.critical(f"❌ LỖI NGHIÊM TRỌNG khi khởi tạo vector store hoặc embedding: {e}")
    traceback.print_exc()
    embeddings = None
    vector_store = None

# --- 3. HÀM XỬ LÝ TÀI LIỆU ---

def load_document(temp_file_path: str, original_filename: str):
    """Tải tài liệu từ file PDF, TXT hoặc định dạng khác."""
    logger.info(f"📄 Đang tải file: {original_filename}")

    try:
        if original_filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif original_filename.lower().endswith(".txt"):
            loader = TextLoader(temp_file_path, encoding="utf-8")
        else:
            loader = UnstructuredFileLoader(temp_file_path)

        docs = loader.load()
        if not docs:
            logger.warning(f"⚠️ File {original_filename} không trích xuất được nội dung (có thể là PDF scan hoặc rỗng).")
        else:
            logger.info(f"✅ Đã tải {len(docs)} tài liệu từ {original_filename}.")
        return docs

    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc file {original_filename}: {e}")
        traceback.print_exc()
        return []

def split_documents(documents: list):
    """Chia tài liệu thành các đoạn nhỏ để embedding."""
    if not documents:
        logger.warning("⚠️ Không có tài liệu nào để chia.")
        return []

    logger.info(f"✂️ Đang chia {len(documents)} tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        add_start_index=True
    )

    splits = text_splitter.split_documents(documents)
    logger.info(f"✅ Đã chia thành {len(splits)} đoạn văn bản.")
    return splits
def add_documents_to_store(documents: list):
    """Thêm các đoạn văn bản đã chia vào ChromaDB."""
    global vector_store
    if not vector_store or not embeddings:
        logger.error("❌ Vector store hoặc Embeddings chưa được khởi tạo. Dừng lại.")
        return

    if not documents:
        logger.warning("⚠️ Không có tài liệu nào để thêm vào vector store.")
        return

    # Lọc bỏ các đoạn rỗng
    non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
    if not non_empty_docs:
        logger.warning("⚠️ Tất cả các đoạn văn bản đều trống. Không tạo embedding.")
        return

    logger.info(f"🧠 Đang thêm {len(non_empty_docs)} đoạn hợp lệ vào ChromaDB...")
    try:
        vector_store.add_documents(non_empty_docs)
        logger.info(f"✅ Thêm thành công! Tổng số vector hiện có: {vector_store._collection.count()}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi thêm tài liệu vào ChromaDB: {e}")
        traceback.print_exc()

async def process_document_background(temp_path: str, original_name: str):
    """Chạy nền để xử lý tài liệu được upload."""
    logger.info(f"🔄 Bắt đầu xử lý nền cho file: {original_name}")
    try:
        docs = await run_in_threadpool(load_document, temp_path, original_name)
        if not docs:
            logger.warning(f"⚠️ Không trích xuất được nội dung từ file {original_name}. Bỏ qua.")
            return

        splits = await run_in_threadpool(split_documents, docs)

        await run_in_threadpool(add_documents_to_store, splits)
        logger.info(f"🎉 Xử lý file {original_name} hoàn tất.")
    except Exception as e:
        logger.error(f"❌ Lỗi chạy nền khi xử lý {original_name}: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"🧹 Đã xóa file tạm: {temp_path}")
