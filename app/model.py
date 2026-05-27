import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Clases en el mismo orden que durante el entrenamiento
CLASSES = ['accidente', 'congestion', 'fluido', 'obras']

# Transform idéntico al usado en test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50_finetuned.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

def predict(image: Image.Image) -> dict:
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    
    return {
        "clase": CLASSES[predicted_idx],
        "confianza": round(probabilities[predicted_idx].item(), 4),
        "probabilidades": {
            cls: round(probabilities[i].item(), 4)
            for i, cls in enumerate(CLASSES)
        }
    }