import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# -------------------- CONFIG --------------------
DATA_DIR = "/home/abdullah/Model Training/StanfordDogs/Images"  # Path to dataset
BATCH_SIZE = 32
NUM_CLASSES = 120
EPOCHS = 15
LR = 0.001
VAL_SPLIT = 0.2
MODEL_PATH = "efficientnet_b3_dog_breed.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------------------- DATA AUGMENTATION --------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------- DATASET --------------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
num_train = int((1 - VAL_SPLIT) * len(full_dataset))
num_val = len(full_dataset) - num_train

train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
val_dataset.dataset.transform = val_transform  # use simpler transforms for val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"📊 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# -------------------- MODEL --------------------
model = models.efficientnet_b3(pretrained=True)

# Unfreeze only last few layers for fine-tuning
for name, param in model.named_parameters():
    if "features.6" in name or "features.7" in name or "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace classification head
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

# -------------------- LOSS, OPTIMIZER, SCHEDULER --------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -------------------- TRAINING LOOP --------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"\n🧠 Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # -------------------- VALIDATION --------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"📈 Validation Accuracy: {val_acc:.2f}%")

    # Step scheduler
    scheduler.step()

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Best model saved with accuracy: {best_val_acc:.2f}%")

print(f"\n✅ Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Model saved as: {MODEL_PATH}")
