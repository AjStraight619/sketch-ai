import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.sketch_cnn import SketchCNN
from models.dataset import SketchDataset

# Settings
DATA_DIR = "data/images"
BATCH_SIZE = 128
VAL_SPLIT = 0.2

# Classes
CLASSES = sorted([f.replace(".npy", "") for f in os.listdir(DATA_DIR) if f.endswith(".npy")])
NUM_CLASSES = len(CLASSES)

# Split data into train/val indices (same as you did for training)
train_idxs = []
val_idxs = []
for class_name in CLASSES:
    imgs = np.load(os.path.join(DATA_DIR, f"{class_name}.npy"))
    num = len(imgs)
    idxs = np.arange(num)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(idxs)
    split = int(num * (1 - VAL_SPLIT))
    train_idxs.append(idxs[:split])
    val_idxs.append(idxs[split:])

# Validation dataset/loader
val_ds = SketchDataset(DATA_DIR, CLASSES, val_idxs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Try to load 'sketchcnn_best.pth' if it exists, else latest epoch
if os.path.exists("sketchcnn_best.pth"):
    checkpoint_path = "sketchcnn_best.pth"
    print("Loading best checkpoint: sketchcnn_best.pth")
else:
    # Find latest epoch checkpoint (e.g., sketchcnn_epoch5.pth)
    all_ckpts = [f for f in os.listdir('.') if f.startswith("sketchcnn_epoch") and f.endswith(".pth")]
    if not all_ckpts:
        raise RuntimeError("No checkpoint found!")
    checkpoint_path = sorted(all_ckpts)[-1]
    print(f"Loading checkpoint: {checkpoint_path}")

model = SketchCNN(NUM_CLASSES)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation loop
correct = 0
total = 0
per_class_correct = np.zeros(NUM_CLASSES, dtype=int)
per_class_total = np.zeros(NUM_CLASSES, dtype=int)

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        # Per-class stats
        for i in range(len(labels)):
            per_class_total[labels[i].item()] += 1
            if preds[i].item() == labels[i].item():
                per_class_correct[labels[i].item()] += 1

val_acc = 100 * correct / total
print(f"Validation accuracy: {val_acc:.2f}%")

# (Optional) Print per-class accuracy for debugging/insight
print("\nPer-class accuracy (top 10 classes):")
for i, cname in enumerate(CLASSES[:10]):
    pct = 100 * per_class_correct[i] / max(1, per_class_total[i])
    print(f"{cname:20}: {pct:.2f}%")

# Save overall results to file for logging
with open("val_results.txt", "w") as f:
    f.write(f"Validation accuracy: {val_acc:.2f}%\n")
    for i, cname in enumerate(CLASSES):
        pct = 100 * per_class_correct[i] / max(1, per_class_total[i])
        f.write(f"{cname:20}: {pct:.2f}%\n")
