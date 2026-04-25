import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Gesture settings
GESTURE_CLASSES = ['help', 'stop', 'yes', 'no', 'calm', 'attention', 'emergency', 'suspicious', 'cancel', 'unknown']
NUM_CLASSES = 10
EPOCHS = 10
def get_latest_model_and_next_version(base_name):
    import glob
    import re
    # Lock to .pth for iterative versions
    pattern = os.path.join(BASE_DIR, f"{base_name}_v*.pth")
    files = glob.glob(pattern)
    if not files:
        original = os.path.join(BASE_DIR, f"{base_name}.pth")
        return original, 2
    
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version, reverse=True)
    latest_file = files[0]
    latest_version = extract_version(latest_file)
    return latest_file, latest_version + 1

ORIGINAL_MODEL, NEXT_VERSION = get_latest_model_and_next_version("gesture_model")

class GestureEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return embedding, logits

class BufferDataset(Dataset):
    def __init__(self, buffer_dir, transform=None):
        self.buffer_dir = buffer_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for i, cls in enumerate(GESTURE_CLASSES):
            cls_dir = os.path.join(buffer_dir, cls)
            if os.path.exists(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.endswith('.jpg') or f.endswith('.png'):
                        self.images.append(os.path.join(cls_dir, f))
                        self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def load_gesture_model():
    if not os.path.exists(ORIGINAL_MODEL):
        raise FileNotFoundError(f"Missing gesture checkpoint: {ORIGINAL_MODEL}")

    model = GestureEncoder(NUM_CLASSES)
    checkpoint = torch.load(ORIGINAL_MODEL, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model

def retrain_gesture_model():
    print("--- Starting GESTURE Continual Fine-Tuning ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gesture_model().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load 15% buffer data. Assume original data handling is skipped or mocked
    buffer_dset = BufferDataset(os.path.join(BASE_DIR, "buffers", "gesture"), transform=transform)
    if len(buffer_dset) == 0:
        raise RuntimeError("No real gesture buffer images found. Refusing to train on dummy data.")
        
    loader = DataLoader(buffer_dset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (imgs, lbls) in enumerate(loader):
            if isinstance(imgs, torch.Tensor):
                imgs, lbls = imgs.to(device), (lbls.to(device) if isinstance(lbls, torch.Tensor) else torch.tensor([lbls]).to(device))
            else:
                continue
            optimizer.zero_grad()
            _, logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/max(1, len(loader)):.4f}")

    # Save Updated Models (Locked to .pth)
    save_path_pth = os.path.join(BASE_DIR, f"gesture_model_v{NEXT_VERSION}.pth")
    torch.save(model.state_dict(), save_path_pth)
    print(f"✓ Saved v{NEXT_VERSION} state_dict to: {save_path_pth}")

    print(f"--- Gesture Fine-Tuning v{NEXT_VERSION} Complete ---")

if __name__ == '__main__':
    retrain_gesture_model()
