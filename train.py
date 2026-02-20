import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# 1. Seed
# ==========================
torch.manual_seed(42)
np.random.seed(42)

# ==========================
# 2. Dataset
# ==========================
class MultiLabelDataset(Dataset):
    def __init__(self, image_folder, label_file):
        self.image_folder = image_folder
        self.data = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                img_path = os.path.join(image_folder, parts[0])
                if os.path.exists(img_path):
                    self.data.append(parts)

        print("Total valid images:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = os.path.join(self.image_folder, row[0])
        image = Image.open(img_path).convert("RGB")

        labels = []
        mask = []

        for value in row[1:]:
            if value == "NA":
                labels.append(0)
                mask.append(0)
            else:
                labels.append(float(value))
                mask.append(1)

        return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ==========================
# 3. Transforms
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# 4. Load Dataset
# ==========================
dataset = MultiLabelDataset("images", "labels.txt")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply transforms dynamically
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

def collate_fn(batch):
    images, labels, masks = [], [], []
    for img, lbl, msk in batch:
        img = train_dataset.dataset.transform(img)
        images.append(img)
        labels.append(lbl)
        masks.append(msk)
    return torch.stack(images), torch.stack(labels), torch.stack(masks)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# ==========================
# 5. Model (Freeze Backbone)
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 4)
)

model = model.to(device)

# ==========================
# 6. Class Weights
# ==========================
pos_counts = np.zeros(4)
neg_counts = np.zeros(4)

for _, labels, mask in train_dataset:
    for i in range(4):
        if mask[i] == 1:
            if labels[i] == 1:
                pos_counts[i] += 1
            else:
                neg_counts[i] += 1

pos_weight = torch.tensor(
    neg_counts / (pos_counts + 1e-6),
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# ==========================
# 7. Training Loop
# ==========================
num_epochs = 15
best_val_loss = float("inf")

train_losses = []
val_losses = []

for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for images, labels, mask in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss = loss * mask
        loss = loss.sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels, mask in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

print("Training finished.")

# ==========================
# 8. Plot Graph
# ==========================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Aimonk_multilabel_problem")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()
