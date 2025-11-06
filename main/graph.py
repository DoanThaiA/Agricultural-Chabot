
import base64
import json
import uuid
from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from agents.predict_image import predict_image_agent
from agents.text_analyzer import text_analyzer_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
from langchain_huggingface import HuggingFaceEmbeddings
from agents.vector_store import vector_store

load_dotenv()
def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class AgricultureState(TypedDict):
    """State definition for the agriculture chatbot"""
    messages: Annotated[List, operator.add]
    user_query: str
    query_type: str
    condensed_query: str
    image_data: Optional[str]
    disease_info: Optional[dict]
    context: dict


def condense_conversation_history(state: AgricultureState) -> AgricultureState:
    """
    Nén lịch sử hội thoại VÀ xử lý thông tin bổ sung.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"condensed_query": "", "messages": [AIMessage(content="Lỗi: Không có tin nhắn.")]}

    user_query = messages[-1].content
    chat_history = messages[-6:-1]

    if not chat_history:
        return {
            **state,
            "condensed_query": user_query,
            "user_query": user_query
        }

    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    prompt = f"""Bạn là một trợ lý AI có nhiệm vụ viết lại câu hỏi của người dùng.
Dựa trên lịch sử hội thoại và câu hỏi mới, hãy làm theo các quy tắc sau:

1.  **Tiếp nối (Follow-up):** Nếu câu hỏi mới là câu hỏi tiếp nối (ví dụ: "chữa thế nào?", "nguyên nhân?"), 
    hãy viết lại nó thành một câu hỏi độc lập, đầy đủ ngữ cảnh từ lịch sử.
    *Ví dụ Lịch sử: "Bệnh X"; Câu mới: "Cách chữa?"; Kết quả: "Cách chữa bệnh X?"*

2.  **Bổ sung (Correction/Addition):** Nếu câu hỏi mới là một thông tin **bổ sung** hoặc **sửa lỗi** cho câu hỏi ngay trước đó 
    (ví dụ: người dùng mô tả triệu chứng, sau đó nói tên cây trồng), 
    hãy **kết hợp** lịch sử gần nhất và câu mới thành một câu hỏi hoàn chỉnh.
    *Ví dụ Lịch sử: "...vết hình thoi"; Câu mới: "trên cây lúa"; Kết quả: "cây lúa có vết hình thoi là bệnh gì?"*
3. **Thay đổi** Nếu người dùng hỏi một câu hỏi hoàn toàn mới không liên quan gì đến tin nhắn trước đó hãy gì nguyên câu hỏi
    của người dùng.Ví dụ khi người dùng hỏi về loại cây trồng khác, hoặc vấn để khác không liên quan đến quá khứ.

Lịch sử:
{history_str}

Câu mới: {user_query}

Kết quả (viết lại hoặc kết hợp):"""

    response = llm.invoke(prompt)
    condensed_query = response.content.strip()

    print(f"[Condenser]: Đã nén thành: {condensed_query}")

    return {
        **state,
        "condensed_query": condensed_query,
        "user_query": user_query
    }
def classify_input(state: AgricultureState) -> AgricultureState:
    """Classify the type of user query"""
    image_data = state.get("image_data")
    if image_data:
        return {
            **state,
            "query_type": "image_disease"
        }
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
    classification_prompt = f"""Truy vấn: {state['condensed_query']}
    Nhiệm vụ của bạn là đọc nội dung người dùng nhập vào và phân loại nó thành đúng một trong ba loại sau đây:
    text_disease: khi người dùng mô tả bằng chữ các triệu chứng, dấu hiệu hoặc tình trạng bệnh của cây trồng và muốn biết đó là bệnh gì, nguyên nhân hoặc cách chữa. Ví dụ như “Lá lúa bị đốm nâu, cây còi cọc là bệnh gì” hoặc “Cây cà chua bị vàng lá, héo dần là sao”.
    normal_qa: khi người dùng hỏi về kiến thức nông nghiệp nói chung, không mô tả bệnh cụ thể. Bao gồm các câu hỏi về kỹ thuật trồng, chăm sóc, bón phân, thời vụ, giống cây, côn trùng, đất đai, hoặc dinh dưỡng. Ví dụ như “Cách bón phân cho cây cam”, “Đất trồng rau nên có độ pH bao nhiêu” hoặc “Giống lúa nào năng suất cao”.
    chitchat: khi người dùng chào hỏi, cảm ơn, nói chuyện phiếm hoặc đặt các câu hỏi không liên quan trực tiếp đến kiến thức nông nghiệp. Nhóm này cũng bao gồm những câu hỏi về thời tiết, giá cả, tin tức hoặc bất kỳ thông tin nào cần tìm kiếm trên mạng. Ví dụ như “Chào bạn”, “Cảm ơn nhé”, “Thời tiết ở Hà Nội hôm nay thế nào” hoặc “Giá phân DAP hôm nay là bao nhiêu”.
    Chỉ trả về **một trong ba nhãn duy nhất** sau đây, không giải thích thêm:
    - text_disease  
    - normal_qa  
    - chitchat"""
    response = llm.invoke([HumanMessage(content=classification_prompt)])
    query_type = response.content.strip().lower()
    valid_types = ["text_disease", "normal_qa", "chitchat"]
    if query_type not in valid_types:
        query_type = "chitchat"
    return {
        **state,
        "query_type": query_type}
def chitchat(state: AgricultureState) -> AgricultureState:
    """Tạo phản hồi nhanh cho các câu chào hỏi, cảm ơn."""
    # Bạn có thể dùng LLM nếu muốn câu trả lời đa dạng
    prompt = f"Người dùng: {state['user_query']}. Bạn là trợ lý AI thân thiện. Hãy trả lời ngắn gọn."
    try:
        # Gọi model hoặc agent
        response = text_analyzer_agent.invoke({"messages": [HumanMessage(content=prompt)]})

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
    """Phân tích ảnh (ĐÃ SỬA: Lưu file tạm, gửi path cho agent)."""

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

        # 5. Chuẩn bị input đơn giản cho Agent (chứa đường dẫn file)
        agent_input_prompt = f"""Hãy phân tích hình ảnh cây trồng tại đường dẫn sau: {temp_filename}

        Sử dụng tool 'predict' để xác định bệnh và độ tin cậy.
        Chỉ trả về kết quả JSON từ tool. Không thêm bất kỳ lời giải thích nào.
        Ví dụ JSON mong muốn:
        {{
          "label": "...", 
          "confidence": 0.xx 
        }}
        """
        agent_input_message = HumanMessage(content=agent_input_prompt)

        # 6. Gọi agent VỚI ĐƯỜNG DẪN FILE
        response = predict_image_agent.invoke({"messages": [agent_input_message]})

        # 7. Xử lý kết quả trả về từ agent
        agent_output = response.get('output') or response['messages'][-1].content

        try:
            # Agent có thể trả về JSON trực tiếp hoặc trong ```json ... ```
            if "```json" in agent_output:
                agent_output = agent_output.split("```json")[1].split("```")[0]

            # Parse kết quả JSON từ tool (do agent trả về)
            tool_result = json.loads(agent_output.strip())

            # Kiểm tra xem tool có trả về lỗi không
            if "error" in tool_result:
                raise ValueError(tool_result["error"])

            # Chuyển đổi định dạng cho phù hợp với AgricultureState
            disease_info = {
                # Cố gắng tách tên cây khỏi tên bệnh nếu có
                "plant_type": tool_result.get("label", "Unknown").split(' ')[0],
                "disease_detected": tool_result.get("label", "Analysis inconclusive"),
                # Chuyển đổi confidence (0.x) thành chuỗi %
                "confidence": f"{tool_result.get('confidence', 0) * 100:.1f}%"
            }

        except Exception as parse_error:
            print(f"Lỗi parse JSON từ agent output: {parse_error}\nRaw output: {agent_output}")
            disease_info = {"plant_type": "Unknown", "disease_detected": "Analysis inconclusive", "confidence": None}

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý ảnh hoặc gọi agent: {e}")
        disease_info = {"plant_type": "Unknown", "disease_detected": "Error processing image", "confidence": None}

    finally:
        # Xóa file tạm sau khi dùng xong
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                print(f"Đã xóa file tạm: {temp_filename}")
            except Exception as delete_error:
                print(f"Lỗi khi xóa file tạm {temp_filename}: {delete_error}")

    return {
        **state,
        "disease_info": disease_info,
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
    global vector_store  # Đảm bảo dùng đúng
    if not vector_store:
        print("Lỗi: vector_store không được load, bỏ qua RAG.")
        return {"context": {"retrieved_docs": [], "sources": [], "has_good_content": False}}
    if state.get('disease_info'):
        search_query = f"{state['disease_info'].get('disease_detected', '')} {state['condensed_query']}"
    else:
        search_query = state['condensed_query']

    docs = vector_store.similarity_search_with_relevance_scores(search_query, k=3)

    # --- Thêm Log để kiểm tra score ---
    print(f"\n--- KẾT QUẢ RAG (Query: {search_query}) ---")
    print(docs)
    # --- Kết thúc Log ---

    good_docs = [doc for doc, score in docs if score > 0.55]
    has_good_context = len(good_docs) > 0
    retrieved_docs_list = []
    sources_list = []
    if has_good_context:
        # Nếu tìm thấy, chỉ lấy tài liệu đầu tiên (tốt nhất)
        retrieved_docs_list = [good_docs[0].page_content]
        sources_list = [good_docs[0].metadata.get("source", "Unknown")]
    context = {
        "retrieved_docs": retrieved_docs_list,
        "sources": sources_list,
        "has_good_context": has_good_context
    }

    print(f"--- Has Good Context: {context['has_good_context']} ---")  # Thêm Log

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
        - Plant Type: {state['disease_info'].get('plant_type', 'Unknown')}
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
        3. **Causes/Conditions of spread:** (If any).
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
def create_agriculture_graph():
    memory = InMemorySaver()
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgricultureState)
    workflow.add_node("condense_history", condense_conversation_history)
    workflow.add_node("classify", classify_input)
    workflow.add_node("chitchat", chitchat)
    workflow.add_node("analyze_image", analyze_image)
    workflow.add_node("request_more_info", request_more_info)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("request_clarification", request_clarification)
    workflow.add_node("diagnose_disease", generate_disease_diagnosis)
    workflow.add_node("normal_qa", generate_normal_qa)
    workflow.set_entry_point("condense_history")
    workflow.add_edge("condense_history", "classify")

    def route_after_classify(state: AgricultureState) -> str:

        if state['query_type'] == "image_disease":
            return "analyze_image"
        if state['query_type'] == "chitchat":
            return "chitchat"
        else:
            return "retrieve_knowledge"

    workflow.add_conditional_edges(
        "classify",
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
            if confidence_str is None:
                # Nếu confidence là None (do lỗi phân tích), coi như độ tin cậy thấp
                print("Confidence là None. Yêu cầu thêm thông tin.")
                return "request_more_info"
            # Chuyển đổi "number%" thành float
            confidence_val = float(confidence_str.replace('%', '').strip())

            if confidence_val < 70:
                print(f"Độ tin cậy thấp ({confidence_val}%), yêu cầu thêm thông tin.")
                return "request_more_info"
            else:
                print(f"Độ tin cậy cao ({confidence_val}%), tiếp tục retrieval.")
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