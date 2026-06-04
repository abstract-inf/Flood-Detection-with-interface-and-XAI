import io
import time
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from torchvision import models, transforms
from PIL import Image

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Architecture Definitions ---
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.classifier(self.features(x))

def get_resnet():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def get_inception():
    model = models.inception_v3(pretrained=False, aux_logits=True, init_weights=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def get_vit():
    model = models.vit_b_16(pretrained=False)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model

# --- Model Registry ---
# Maps exact dropdown names to (architecture_function, expected_image_size)
model_registry = {
    "Custom_CNN": (CustomCNN, 224),
    "ResNet_50": (get_resnet, 224),
    "Inception_V3": (get_inception, 299),
    "ViT_B-16": (get_vit, 224)
}

loaded_models = {}

def load_model(name):
    if name in loaded_models:
        return loaded_models[name]
    
    model_fn, img_size = model_registry[name]
    model = model_fn()
    
    # Path depends on where your models are saved. Adjust this relative path if needed.
    path = f"./saved_models/{name}_best.pt" 
    
    state_dict = torch.load(path, map_location=device, weights_only=False)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device).eval()
    
    loaded_models[name] = (model, img_size)
    print(f"Loaded {name} into memory.")
    return loaded_models[name]


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    model, img_size = load_model(model_name)
    
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = tf(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        out = model(tensor)
        end_time = time.time()
        
        pred_idx = torch.argmax(out, dim=1).item()
        prob = torch.softmax(out, dim=1)[0][pred_idx].item()
        
    class_names = ["Flood", "Not_Flood"]
    
    return {
        "prediction": class_names[pred_idx], 
        "confidence": prob,
        "inference_time_ms": (end_time - start_time) * 1000
    }