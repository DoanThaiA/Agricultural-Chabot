
import base64
import uuid
from typing import TypedDict, Annotated, List, Optional, Literal
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from agents.predict_image import predict
from langgraph.checkpoint.memory import InMemorySaver
import os
from agents.vector_store import vector_store
from pydantic import BaseModel, Field
load_dotenv()
reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class QueryAnalysis(BaseModel):
    """Phân tích câu hỏi người dùng: Viết lại câu hỏi và Phân loại chủ đề."""
    condensed_query: str = Field(..., description="Câu hỏi đã được viết lại cho rõ nghĩa dựa trên lịch sử hội thoại.")
    query_type: Literal["text_disease", "normal_qa", "chitchat"] = Field(..., description="Loại câu hỏi: text_disease (bệnh cây), normal_qa (hỏi đáp chung), chitchat (xã giao).")
class AgricultureState(TypedDict):
    """State definition for the agriculture chatbot"""
    messages: Annotated[List, operator.add]
    user_query: str
    query_type: str
    condensed_query: str
    image_data: Optional[str]
    disease_info: Optional[dict]
    context: dict


# def cosine_similarity(a, b):
#     """Tính cosine similarity giữa 2 vector numpy."""
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# def condense_conversation_history(state: AgricultureState) -> AgricultureState:
#     """
#     Nén lịch sử hội thoại VÀ xử lý thông tin bổ sung.
#     """
#     llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
#     messages = state.get("messages", [])
#     if not messages:
#         return {"condensed_query": "", "messages": [AIMessage(content="Lỗi: Không có tin nhắn.")]}
#
#     user_query = messages[-1].content
#     chat_history = messages[-6:-1]
#     history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
#     if state.get("image_data"):
#         return {
#             **state,
#             "condensed_query": user_query,
#             "user_query": user_query
#         }
#     if not chat_history:
#         return {
#             **state,
#             "condensed_query": user_query,
#             "user_query": user_query
#         }
#
#     try:
#         query_emb = embeddings_model.embed_query(user_query)
#         history_embs = [embeddings_model.embed_query(msg.content) for msg in chat_history]
#         similarities = [cosine_similarity(query_emb, emb) for emb in history_embs]
#
#         max_sim = max(similarities)
#         print(f"[Embedding Similarity] Mức liên quan cao nhất: {max_sim:.3f}")
#
#     except Exception as e:
#         print(f"Lỗi khi tính embedding similarity: {e}")
#         max_sim = 0.0
#     if max_sim < 0.4:
#         return {
#             **state,
#             "condensed_query": user_query,
#             "user_query": user_query
#         }
#     elif max_sim > 0.6:
#         prompt = f"""Bạn là một trợ lý AI có nhiệm vụ viết lại câu hỏi của người dùng.
#     Dựa trên lịch sử hội thoại và câu hỏi mới, hãy làm theo các quy tắc sau:
#
#     1.  **Tiếp nối (Follow-up):** Nếu câu hỏi mới là câu hỏi tiếp nối (ví dụ: "chữa thế nào?", "nguyên nhân?"),
#         hãy viết lại nó thành một câu hỏi độc lập, đầy đủ ngữ cảnh từ lịch sử.
#         *Ví dụ Lịch sử: "Bệnh X"; Câu mới: "Cách chữa?"; Kết quả: "Cách chữa bệnh X?"*
#
#     2.  **Bổ sung (Correction/Addition):** Nếu câu hỏi mới là một thông tin **bổ sung** hoặc **sửa lỗi** cho câu hỏi ngay trước đó
#         (ví dụ: người dùng mô tả triệu chứng, sau đó nói tên cây trồng),
#         hãy **kết hợp** lịch sử gần nhất và câu mới thành một câu hỏi hoàn chỉnh.
#         *Ví dụ Lịch sử: "...vết hình thoi"; Câu mới: "trên cây lúa"; Kết quả: "cây lúa có vết hình thoi là bệnh gì?"*
#             Lịch sử: {history_str}
#
#             Câu mới: {user_query}
#             Kết quả:"""
#         response = llm.invoke(prompt)
#         condensed_query = response.content.strip()
#         return {
#             **state,
#             "condensed_query": condensed_query,
#             "user_query": user_query
#         }
#
#
#     else:
#         prompt = f"""Bạn là một trợ lý AI có nhiệm vụ viết lại câu hỏi của người dùng.
#     Dựa trên lịch sử hội thoại và câu hỏi mới, hãy làm theo các quy tắc sau:
#
#     1.  **Tiếp nối (Follow-up):** Nếu câu hỏi mới là câu hỏi tiếp nối (ví dụ: "chữa thế nào?", "nguyên nhân?"),
#         hãy viết lại nó thành một câu hỏi độc lập, đầy đủ ngữ cảnh từ lịch sử.
#         *Ví dụ Lịch sử: "Bệnh X"; Câu mới: "Cách chữa?"; Kết quả: "Cách chữa bệnh X?"*
#
#     2.  **Bổ sung (Correction/Addition):** Nếu câu hỏi mới là một thông tin **bổ sung** hoặc **sửa lỗi** cho câu hỏi ngay trước đó
#         (ví dụ: người dùng mô tả triệu chứng, sau đó nói tên cây trồng),
#         hãy **kết hợp** lịch sử gần nhất và câu mới thành một câu hỏi hoàn chỉnh.
#         *Ví dụ Lịch sử: "...vết hình thoi"; Câu mới: "trên cây lúa"; Kết quả: "cây lúa có vết hình thoi là bệnh gì?"*
#     3. **Thay đổi** Nếu người dùng hỏi một câu hỏi hoàn toàn mới không liên quan gì đến tin nhắn trước đó hãy gì nguyên câu hỏi
#         của người dùng.Ví dụ khi người dùng hỏi về loại cây trồng khác, hoặc vấn để khác không liên quan đến quá khứ.
#
#     Lịch sử:
#     {history_str}
#
#     Câu mới: {user_query}
#
#     Kết quả (viết lại hoặc kết hợp):"""
#
#         response = llm.invoke(prompt)
#         condensed_query = response.content.strip()
#
#         print(f"[Condenser]: Đã nén thành: {condensed_query}")
#
#         return {
#             **state,
#             "condensed_query": condensed_query,
#             "user_query": user_query
#         }
# def classify_input(state: AgricultureState) -> AgricultureState:
#     """Classify the type of user query"""
#     image_data = state.get("image_data")
#     if image_data:
#         return {
#             **state,
#             "query_type": "image_disease"
#         }
#     llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
#     classification_prompt = f"""Truy vấn của người dùng: {state['condensed_query']}
#
# Nhiệm vụ: Phân loại truy vấn thành **một và chỉ một** trong ba nhãn sau. Hãy đọc kỹ nội dung để xác định đúng chủ đích.
#
# 1. text_disease
#    - Khi người dùng mô tả **triệu chứng thực tế** trên cây, lá, thân, rễ, quả…
#    - Thường xuất hiện các mô tả như: đốm lá, vàng lá, héo rũ, cháy mép, nấm, thối rễ…
#    - Mục đích chính: **nhận diện bệnh hoặc vấn đề cụ thể của cây dựa trên triệu chứng.**
#
# 2. normal_qa
#    - Khi người dùng hỏi về **kiến thức nông nghiệp chung**, không nhằm mô tả triệu chứng để nhận dạng bệnh.
#    - Bao gồm: nguyên nhân, cách chăm sóc, cách phòng bệnh, quy trình trồng, dinh dưỡng, giá nông sản, tác hại của bệnh, thuốc trị, kỹ thuật canh tác, tư vấn giống…
#    - Không kèm mô tả triệu chứng thực tế.
#
# 3. chitchat
#    - Khi người dùng giao tiếp xã giao hoặc nội dung **không liên quan đến nông nghiệp**.
#    - Ví dụ: chào hỏi, cảm ơn, khen/chê, hỏi chuyện cá nhân, nói linh tinh…
#
#  Chỉ trả về **một trong ba nhãn duy nhất** dưới đây, không giải thích thêm:
# - text_disease
# - normal_qa
# - chitchat"""
#     response = llm.invoke([HumanMessage(content=classification_prompt)])
#     query_type = response.content.strip().lower()
#     valid_types = ["text_disease", "normal_qa", "chitchat"]
#     if query_type not in valid_types:
#         query_type = "chitchat"
#     return {
#         **state,
#         "query_type": query_type}
def process_user_query(state: AgricultureState) -> AgricultureState:
    """
 nén lịch sử vừa phân loại .
    """

    image_data = state.get("image_data")
    if image_data:
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        return {
            **state,
            "condensed_query": user_query,
            "user_query": user_query,
            "query_type": "image_disease",
            "disease_info": None
        }

    # 2. Chuẩn bị dữ liệu cho LLM
    messages = state.get("messages", [])
    if not messages:
        return {**state, "condensed_query": "", "query_type": "chitchat"}

    user_query = messages[-1].content

    chat_history = messages[-6:-1]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

    structured_llm = llm.with_structured_output(QueryAnalysis)

    system_prompt = f"""Bạn là một chuyên gia AI về nông nghiệp. Nhiệm vụ của bạn là xử lý câu hỏi của người dùng dựa trên lịch sử hội thoại.

    LỊCH SỬ HỘI THOẠI:
    {history_str}

    CÂU HỎI MỚI: {user_query}
** Nhiệm vụ 1:
#     Dựa trên lịch sử hội thoại và câu hỏi mới, hãy làm theo các quy tắc sau:
#
#     1.  **Tiếp nối (Follow-up):** Nếu câu hỏi mới là câu hỏi tiếp nối (ví dụ: "chữa thế nào?", "nguyên nhân?"),
#         hãy viết lại nó thành một câu hỏi độc lập, đầy đủ ngữ cảnh từ lịch sử.
#         *Ví dụ Lịch sử: "Bệnh X"; Câu mới: "Cách chữa?"; Kết quả: "Cách chữa bệnh X?"*
#
#     2.  **Bổ sung (Correction/Addition):** Nếu câu hỏi mới là một thông tin **bổ sung** hoặc **sửa lỗi** cho câu hỏi ngay trước đó
#         (ví dụ: người dùng mô tả triệu chứng, sau đó nói tên cây trồng),
#         hãy **kết hợp** lịch sử gần nhất và câu mới thành một câu hỏi hoàn chỉnh.
#         *Ví dụ Lịch sử: "...vết hình thoi"; Câu mới: "trên cây lúa"; Kết quả: "cây lúa có vết hình thoi là bệnh gì?"*
#     3. **Thay đổi** Nếu người dùng hỏi một câu hỏi hoàn toàn mới không liên quan gì đến tin nhắn trước đó hãy gì nguyên câu hỏi
#         của người dùng.Ví dụ khi người dùng hỏi về loại cây trồng khác, hoặc vấn để khác không liên quan đến quá khứ.
#     Kết quả (viết lại hoặc kết hợp):
**Nhiệm vụ 2:
    Phân loại truy vấn thành **một và chỉ một** trong ba nhãn sau. Hãy đọc kỹ nội dung để xác định đúng chủ đích.
#
# 1. text_disease
#    - Khi người dùng mô tả **triệu chứng thực tế** trên cây, lá, thân, rễ, quả…
#    - Thường xuất hiện các mô tả như: đốm lá, vàng lá, héo rũ, cháy mép, nấm, thối rễ…
#    - Mục đích chính: **nhận diện bệnh hoặc vấn đề cụ thể của cây dựa trên triệu chứng.**
#
# 2. normal_qa
#    - Khi người dùng hỏi về **kiến thức nông nghiệp chung**, không nhằm mô tả triệu chứng để nhận dạng bệnh.
#    - Bao gồm: nguyên nhân, cách chăm sóc, cách phòng bệnh, quy trình trồng, dinh dưỡng, giá nông sản, tác hại của bệnh, thuốc trị, kỹ thuật canh tác, tư vấn giống…
#    - Không kèm mô tả triệu chứng thực tế.
#
# 3. chitchat
#    - Khi người dùng giao tiếp xã giao hoặc nội dung **không liên quan đến nông nghiệp**.
#    - Ví dụ: chào hỏi, cảm ơn, khen/chê, hỏi chuyện cá nhân, nói linh tinh…
#
    """

    try:
        # Gọi LLM 1 lần duy nhất
        result = structured_llm.invoke(system_prompt)

        return {
            **state,
            "condensed_query": result.condensed_query,
            "query_type": result.query_type,
            "user_query": user_query
        }

    except Exception as e:
        print(f"Lỗi khi xử lý query (fallback về normal_qa): {e}")
        # Fallback an toàn nếu API lỗi
        return {
            **state,
            "condensed_query": user_query,
            "query_type": "normal_qa",  # Mặc định coi là câu hỏi thường
            "user_query": user_query
        }
def chitchat(state: AgricultureState) -> AgricultureState:
    """Tạo phản hồi nhanh cho các câu chào hỏi, cảm ơn."""
    # Bạn có thể dùng LLM nếu muốn câu trả lời đa dạng
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

    prompt = f"Người dùng: {state['user_query']}. Bạn là trợ lý nông nghiệp thân thiện Hãy trả lời ngắn gọn."
    try:

        response = llm.invoke([SystemMessage("Bạn là trợ lý nông nghiệp thân thiện"),HumanMessage(content=prompt)])

        # Xử lý kết quả trả về
        if isinstance(response, dict):
            # Trường hợp model trả về dạng dict
            if "output" in response:
                content = response["output"]
            elif "messages" in response and response["messages"]:
                content = response["messages"][-1].content
            else:
                content = str(response)
        elif hasattr(response, "content"):
            # Trường hợp là AIMessage
            content = response.content
        else:
            # Nếu chỉ là chuỗi hoặc object khác
            content = str(response)

    except Exception as e:
        print(f"[Chitchat Error]: {e}")
        content = "Xin lỗi, tôi đang gặp sự cố. Bạn thử lại sau nhé 🌱"

    # Trả về đúng dạng dict mà LangGraph yêu cầu
    return {
        "messages": [AIMessage(content=content)]
    }


def analyze_image(state: AgricultureState) -> AgricultureState:
    """Phân tích ảnh"""

    image_data = state.get('image_data')  # Lấy dữ liệu base64
    temp_filename = None  # Khởi tạo để dùng trong finally

    if not image_data:
        return {
            "disease_info": {"error": "No image provided"},
            "messages": [AIMessage(content="Không có ảnh nào được gửi lên.")]
        }

    try:
        # 1. Decode base64 thành bytes
        image_bytes = base64.b64decode(image_data)

        # 2. Tạo thư mục tạm nếu chưa có
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # 3. Tạo tên file tạm ngẫu nhiên
        temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")

        # 4. Lưu ảnh vào file tạm
        with open(temp_filename, "wb") as f:
            f.write(image_bytes)
        print(f"Ảnh tạm đã được lưu tại: {temp_filename}")
        response = predict(image_path=temp_filename)


        disease_info = {
            "plant_type": "Cây",
            "disease_detected": response.get("label", "Unknown"),
            "confidence": f"{response.get('confidence', 0) * 100:.1f}%"
            }

    finally:
        # Xóa file tạm sau khi dùng xong
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                print(f"Đã xóa file tạm: {temp_filename}")
            except Exception as delete_error:
                print(f"Lỗi khi xóa file tạm {temp_filename}: {delete_error}")

    return {
        "disease_info": disease_info
    }


def request_more_info(state: AgricultureState) -> AgricultureState:
    """Tạo một tin nhắn yêu cầu người dùng gửi lại ảnh hoặc tin nhắn cung cấp thêm thông tin"""
    confidence_str = state['disease_info'].get('confidence', '0%')
    disease_detected = state['disease_info'].get('disease_detected', 'không xác định')
    message_content = f"""Kết quả phân tích ảnh có độ tin cậy hơi thấp ({confidence_str} cho bệnh {disease_detected}).
    Để chẩn đoán chính xác hơn, bạn vui lòng:
    1.  **Gửi một bức ảnh khác** (rõ nét hơn, đủ sáng, chụp gần khu vực bị bệnh).
    2.  **Hoặc mô tả thêm** về các triệu chứng bạn quan sát được"""
    return {
        "messages": [AIMessage(content=message_content)]
    }


def retrieve_knowledge(state: AgricultureState) -> AgricultureState:
    global vector_store, reranker_model
    if not vector_store:
        print("Lỗi: vector_store không được load, bỏ qua RAG.")
        return {"context": {"retrieved_docs": [], "sources": [], "has_good_content": False}}
    if state.get('disease_info'):
        search_query = f"{state['disease_info'].get('disease_detected', '')} {state['condensed_query']}"
    else:
        search_query = state['condensed_query']

    try:
        initial_docs = vector_store.similarity_search(search_query, k=2)
    except Exception as e:
        print(f"Lỗi Vector Search: {e}")
        initial_docs = []
    final_docs = []
    if initial_docs:
        pairs = [[search_query, doc.page_content] for doc in initial_docs]
        scores = reranker_model.predict(pairs)
        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        RERANK_THRESHOLD = 0.0
        valid_docs = []
        for doc, score in scored_docs:
            print(f"Score: {score:.4f} | Source: {doc.metadata.get('source', 'Unknown')}")
            if score > RERANK_THRESHOLD:
                valid_docs.append(doc)
        if len(valid_docs) > 0:
            final_docs = valid_docs[:1]
        else:
            final_docs = []
    retrieved_contents = [doc.page_content for doc in final_docs]
    sources_list = [doc.metadata.get("source", "Local DB") for doc in final_docs]
    if not final_docs:
        try:
            tavily_tool = TavilySearchResults(max_results=1)
            web_results = tavily_tool.run(search_query)
            if isinstance(web_results, list):
                for res in web_results:
                    content = res.get('content', '')
                    url = res.get('url', 'Web')
                    retrieved_contents.append(f"[Web Search]: {content}")
                    sources_list.append(url)
        except Exception as e:
            print(f"Lỗi Tavily: {e}")
    has_good_context = len(retrieved_contents) > 0
    context = {
        "retrieved_docs": retrieved_contents,
        "sources": sources_list,
        "has_good_context": has_good_context
    }

    print(f"--- Has Good Context: {context['has_good_context']} ---")

    return {
        **state,
        "context": context}


def request_clarification(state: AgricultureState) -> AgricultureState:
    """
    Tạo tin nhắn khi RAG không tìm thấy thông tin liên quan.
    Yêu cầu người dùng mô tả lại.
    """
    message_content = f"""Rất tiếc, tôi không thể tìm thấy thông tin chính xác về "{state['condensed_query']}" trong cơ sở kiến thức của mình.

Bạn có thể vui lòng:
1.  **Mô tả lại các triệu chứng** bằng từ ngữ khác?
2.  **Kiểm tra lại tên** của loại bệnh/cây bạn đang hỏi?

Điều này sẽ giúp tôi tìm kiếm chính xác hơn."""

    return {
        "messages": [AIMessage(content=message_content)]
    }


async def generate_disease_diagnosis(state: AgricultureState) -> AgricultureState:
    """Generate detailed disease diagnosis"""
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0.3)
    context_text = "\n\n".join(state['context'].get('retrieved_docs', []))
    if state['query_type'] == "image_disease":
        disease_context = f"""
        Image Analysis Results:
        - Disease: {state['disease_info'].get('disease_detected', 'Unknown')}
        - Confidence: {state['disease_info'].get('confidence', 'Unknown')}
        """
    else:
        disease_context = f"User's description: {state['condensed_query']}"
    diagnosis_prompt = f"""You are an agricultural consultant. Based on the following information, make a diagnosis of the plant's condition.

    {disease_context}

    Relevant Knowledge:
    {context_text}
    
    Please state clearly:
        1. **Diagnosis:** Disease name and level of confidence.
        2. **Description of symptoms:** (If available in knowledge).
        3.Just extracting words from Relevant Knowledge does not take fabricated information
        4. End with a word of encouragement and an offer of additional support.
        Answer in Vietnamese"""
    try:
        response = await  llm.ainvoke([HumanMessage(content=diagnosis_prompt)])
        final_response_content = response.content.strip()

    # Nếu không có nội dung (ví dụ lỗi)
        if not final_response_content:
            final_response_content = "Xin lỗi, tôi chưa thể tạo câu trả lời lúc này."

    except Exception as e:
        print(f"Lỗi invoke format: {e}")  # Sửa tên lỗi
        final_response_content = "Lỗi khi tạo phản hồi cuối cùng."


    return {
        **state,
        "messages": [AIMessage(content=final_response_content)]
    }


async def generate_normal_qa(state: AgricultureState) -> AgricultureState:
    """Generate response for normal agriculture question"""
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0.3)
    normal_prompt = f"""You are an expert agricultural advisor. Answer the following question comprehensively.
    Question: {state['condensed_query']}
    Please answer accurately and according to the user's request, do not reply to another topic by mistake. Answer in Vietnamese"""
    try:
        response = await  llm.ainvoke([HumanMessage(content=normal_prompt)])
        final_response_content = response.content.strip()

        # Nếu không có nội dung
        if not final_response_content:
            final_response_content = "Xin lỗi, tôi chưa thể tạo câu trả lời lúc này."

    except Exception as e:
        print(f"Lỗi invoke format: {e}")  # Sửa tên lỗi
        final_response_content = "Lỗi khi tạo phản hồi cuối cùng."

    return {
        **state,
        "messages": [AIMessage(content=final_response_content)]
    }
def create_agriculture_graph():
    memory = InMemorySaver()
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgricultureState)
    # workflow.add_node("condense_history", condense_conversation_history)
    # workflow.add_node("classify", classify_input)
    workflow.add_node("process_user_query",process_user_query)
    workflow.add_node("chitchat", chitchat)
    workflow.add_node("analyze_image", analyze_image)
    workflow.add_node("request_more_info", request_more_info)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("request_clarification", request_clarification)
    workflow.add_node("diagnose_disease", generate_disease_diagnosis)
    workflow.add_node("normal_qa", generate_normal_qa)
    # workflow.set_entry_point("condense_history")
    # workflow.add_edge("condense_history", "classify")
    workflow.set_entry_point("process_user_query")

    def route_after_classify(state: AgricultureState) -> str:

        if state['query_type'] == "image_disease":
            return "analyze_image"
        if state['query_type'] == "chitchat":
            return "chitchat"
        else:
            return "retrieve_knowledge"

    workflow.add_conditional_edges(
        "process_user_query",
        route_after_classify,
        {
            "analyze_image": "analyze_image",
            "retrieve_knowledge": "retrieve_knowledge",
            "chitchat": "chitchat"
        }
    )

    def check_confidence(state: AgricultureState) -> str:
        """
        Kiểm tra độ tin cậy từ node analyze_image.
        Chỉ được gọi nếu query_type là 'image_disease'.
        """
        try:
            confidence_str = state['disease_info'].get('confidence', '0%')
            disease = state['disease_info'].get('disease_detected', 'Unknown')

            if confidence_str is None:
                # Nếu confidence là None (do lỗi phân tích), coi như độ tin cậy thấp
                print("Confidence là None. Yêu cầu thêm thông tin.")
                return "request_more_info"
            # Chuyển đổi "number%" thành float
            confidence_val = float(confidence_str.replace('%', '').strip())

            if confidence_val < 70:
                print(f"Phát hiện bệnh {disease}.Độ tin cậy thấp ({confidence_val}%), yêu cầu thêm thông tin.")
                return "request_more_info"
            else:
                print(f"Phát hiện bệnh {disease}.Độ tin cậy cao ({confidence_val}%), tiếp tục truy xuất.")
                return "retrieve_knowledge"
        except Exception as e:
            # Bắt các lỗi khác (ví dụ: không thể chuyển 'number%' thành float)
            print(f"Lỗi khi kiểm tra độ tin cậy: {e}. Yêu cầu thêm thông tin.")
            return "request_more_info"

    workflow.add_conditional_edges(
        "analyze_image",
        check_confidence,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "request_more_info": "request_more_info"
        }
    )

    def route_after_retrieval(state: AgricultureState) -> str:
        if not state['context'].get('has_good_context'):
            print("RAG không tìm thấy context tốt. Yêu cầu làm rõ.")
            return "request_clarification"
        if state['query_type'] in ["image_disease", "text_disease"]:
            return "diagnose_disease"
        else:
            return "normal_qa"

    workflow.add_conditional_edges(
        "retrieve_knowledge",
        route_after_retrieval,
        {
            "diagnose_disease": "diagnose_disease",
            "normal_qa": "normal_qa",
            "request_clarification": "request_clarification"
        }
    )

    workflow.add_edge("diagnose_disease",END)
    workflow.add_edge("normal_qa",END)
    workflow.add_edge("request_more_info", END)
    workflow.add_edge("request_clarification", END)
    workflow.add_edge("chitchat", END)

    return workflow.compile(checkpointer=memory)


app = create_agriculture_graph()
