
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import copy
import time

# --- 設定 ---
DATA_DIR = "/mnt/c/Users/Yang Tan/Desktop/ppe_dataset"
MODEL_SAVE_DIR = "./models/ppe_classifiers"
IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 25 
LEARNING_RATE = 0.001

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# --- 定義極簡 CNN 模型 ---
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        # Input: 3 x 64 x 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 2 classes: Fail(0), Pass(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 資料集類 ---
class PPEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 0: fail, 1: pass
        classes = {'fail': 0, 'pass': 1}
        
        for cls_name, cls_idx in classes.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir): continue
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 訓練函式 ---
def train_model(category_name):
    print(f"\n[{category_name.upper()}] Starting training...")
    
    train_dir = os.path.join(DATA_DIR, category_name)
    if not os.path.exists(train_dir):
        print(f"Error: Directory {train_dir} not found.")
        return

    # 資料增強
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PPEDataset(train_dir, transform=transform)
    
    if len(dataset) == 0:
        print(f"No images found for {category_name}!")
        return

    # 簡單起見，用全量資料訓練 (因為資料很少，主要為了過擬合這批特定場景)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MiniCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    best_loss = 100.0
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # 儲存模型
    save_path = os.path.join(MODEL_SAVE_DIR, f"{category_name}_net.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model("buckle")
    train_model("vest")
