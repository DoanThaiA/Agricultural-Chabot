import torch
from torchvision import transforms
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from torchvision import models
from langchain_cohere import ChatCohere
import torch.nn as nn

load_dotenv()
class_names = {0: 'Bệnh bạc lá cây lúa',
 1: 'Bệnh cháy lá cây ngô phía Bắc',
 2: 'Bệnh nấm phấn trắng trên cây bí',
 3: 'Nhện đỏ hai đốm cây cà chua',
 4: 'Virus vàng xoăn lá cây cà chua',
 5: 'bệnh cháy lá  cây lúa',
 6: 'bệnh cháy lá sớm trên cây cà chua',
 7: 'bệnh ghẻ trên cây táo',
 8: 'bệnh gỉ sắt trên cây ngô',
 9: 'bệnh mốc sướng sớm cây khoai tây',
 10: 'bệnh phấn trắng cây chery',
 11: 'bệnh sương mai cây khoai tây',
 12: 'bệnh thối đen cây nho',
 13: 'bệnh đạo ôn cây lúa',
 14: 'bệnh đốm lá Septoria cây cà chua',
 15: 'bệnh đốm lá xám cây ngô',
 16: 'bệnh đốm nâu trên  cây lúa',
 17: 'bọ cánh cứng gây hại cho cây lúa',
 18: 'cháy bìa lá  cây lúa',
 19: 'cháy lá cây dâu tây',
 20: 'cây cà chua khỏe mạnh',
 21: 'cây dâu tây lành mạnh',
 22: 'cây khoai tây khỏe mạnh',
 23: 'cây lúa khỏe mạnh',
 24: 'cây mâm xôi khỏe mạnh',
 25: 'cây ngô khỏe mạnh',
 26: 'cây nho khỏe mạnh',
 27: 'cây táo khỏe mạnh',
 28: 'cây việt quất khỏe mạnh',
 29: 'cây đào khỏe mạnh',
 30: 'cây đậu nành khỏe mạnh',
 31: 'cây ớt chuông khỏe mạnh',
 32: 'nấm lá cây cà chua',
 33: 'quả chery khỏe mạnh',
 34: 'rỉ táo tuyết trùng cây táo',
 35: 'sởi đen cây nho',
 36: 'thối đen trên cây táo',
 37: 'virus khảm cây cà chua',
 38: 'vàng lá gân xanh cây cam',
 39: 'Đốm mục tiêu trên cây cà chua',
 40: 'đốm lá cây nho',
 41: 'đốm vi khuẩn cây cà chua',
 42: 'đốm vi khuẩn cây đào',
 43: 'đốm vi khuẩn cây ớt chuông',
 44: 'ốc sương trên cây cà chua'}
MODEL_PATH = r"C:\Laptrinhweb\32_Thai\pythonProject\model\plants_disease_model.pth"
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512,len(class_names))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
@tool(description="Used when users send photos")
def predict(image_path:str):
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output,dim = 1)
        conf, pred = torch.max(probs, dim = 1)
    result ={
        "label": class_names[pred.item()],
        "confidence": conf.item()
    }
    return result

llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
tools = [predict]
prompt = ("You are the agent to analyze images, use the predict tool when users send photos")
predict_image_agent = create_react_agent(model =llm, tools = tools,prompt = prompt)
