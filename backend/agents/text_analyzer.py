from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_cohere import ChatCohere
from langchain_tavily import TavilySearch
load_dotenv()

llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
@tool(description="Get detailed weather info at a location")
def get_weather(location: str | None = None):
    if not location:
        return "Bạn muốn xem thời tiết ở đâu? VD:Hà Nội, Hồ Chí Minh"

    if not isinstance(location, str) or not location.strip():
        return "❌ Bạn chưa cung cấp địa điểm hợp lệ để xem thời tiết."

    try:
        resp = requests.get(
            f"https://wttr.in/{location}?format=3",
            headers={"User-Agent": "LangGraphBot/1.0"},
            timeout=5
        )
        if resp.status_code != 200:
            return f"⚠️ Không thể lấy thông tin thời tiết ({resp.status_code})"
        return f"🌤️ {resp.text}"
    except requests.RequestException as e:
        return f"⚠️ Lỗi mạng: {e}"

search_tool = TavilySearch(
    max_results= 2,
    topic="general"
)
tools = [get_weather, search_tool]
prompt = """
Bạn là trợ lý nông nghiệp thông minh, giúp người dùng tra cứu:
-  Thông tin bệnh cây từ tài liệu `plant.json`
- Thời tiết bằng `get_weather`
-  Kiến thức chung bằng `search_tool`

Hướng dẫn:
- Nếu người dùng hỏi thời tiết  dùng `get_weather`.
- Nếu người dùng hỏi ngoài phạm vi trên  dùng `search_tool`.

Trả lời bằng tiếng Việt, thân thiện, ngắn gọn.
"""
text_analyzer_agent = create_react_agent(llm,tools = tools,prompt=prompt)
