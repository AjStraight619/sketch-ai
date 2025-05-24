import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sketch_cnn import SketchCNN
from models.dataset import SketchDataset
import torchvision.transforms as T

# --- Configuration ---
DATA_DIR = "data/images"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 128
EPOCHS = 30
VAL_SPLIT = 0.2
SEED = 42

# --- Create checkpoints directory if needed ---
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Prepare classes ---
CLASSES = sorted([f.replace(".npy", "") for f in os.listdir(DATA_DIR) if f.endswith(".npy")])
NUM_CLASSES = len(CLASSES)

# --- Reproducibility ---
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Split indices for train/val ---
train_idxs, val_idxs = [], []
for class_name in CLASSES:
    imgs = np.load(os.path.join(DATA_DIR, f"{class_name}.npy"))
    num = len(imgs)
    idxs = np.arange(num)
    np.random.shuffle(idxs)
    split = int(num * (1 - VAL_SPLIT))
    train_idxs.append(idxs[:split])
    val_idxs.append(idxs[split:])

# --- Data augmentation for training set ---
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomAffine(degrees=15, translate=(0.12, 0.12), scale=(0.85, 1.15)),
    T.ToTensor()
])
val_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

# --- Build datasets/loaders ---
train_ds = SketchDataset(DATA_DIR, CLASSES, train_idxs, transform=train_transform)
val_ds = SketchDataset(DATA_DIR, CLASSES, val_idxs, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Model and optimizer ---
model = SketchCNN(NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3, verbose=True)

# --- Training loop with validation and checkpointing ---
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(train_ds)

    # --- Validation loop ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1:02d}: Train Loss = {avg_loss:.4f} | Validation Accuracy = {val_acc:.2f}%")

    # Learning rate schedule
    scheduler.step(val_acc)

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "sketchcnn_best.pth")
        )
        print("  Saved new best model checkpoint!")

    # Save every epoch as well (optional)
    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, f"sketchcnn_epoch{epoch+1}.pth")
    )
