import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

# -------------------- CONFIG --------------------
MODEL_PATH = "efficientnet_b3_dog_breed.pth"
DATA_DIR = "/home/abdullah/Model Training/StanfordDogs/Images"  # same as training dir
NUM_CLASSES = 120
IMAGE_PATH = "/home/abdullah/Model Training/images (2).jpeg"  # 🔹 path to your test image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------- LOAD MODEL --------------------
model = models.efficientnet_b3(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -------------------- CLASS NAMES --------------------
# Using folder names as class labels
class_names = os.listdir(DATA_DIR)
class_names.sort()  # ensure same order as during training

# -------------------- PREDICTION FUNCTION --------------------
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]

    print(f"🐶 Predicted Breed: {pred_class}")
    return pred_class

# -------------------- RUN TEST --------------------
predict_image(IMAGE_PATH)
