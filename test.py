import torch

MODEL_PATH = r"C:\Laptrinhweb\32_Thai\pythonProject\backend\model\disease_model.pth"

state_dict = torch.load(MODEL_PATH, map_location="cpu")

for k in state_dict.keys():
    print(k)