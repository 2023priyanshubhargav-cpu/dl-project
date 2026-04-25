import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2

# ==============================================================
# CONTINUAL LEARNING - EMOTION MODEL FINE-TUNING
# Description: Mixes 15% patient buffer data with 85% original data
# ==============================================================

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 1e-4 # Low learning rate to prevent catastrophic forgetting
BATCH_SIZE = 32
NEW_DATA_RATIO = 0.15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_latest_model_and_next_version(base_name):
    import glob
    import re
    pattern = os.path.join(BASE_DIR, f"{base_name}_v*.pt")
    files = glob.glob(pattern)
    if not files:
        # Fallback to original pth or first scripted pt
        original = os.path.join(BASE_DIR, f"{base_name}_full.pth")
        if not os.path.exists(original):
            original = os.path.join(BASE_DIR, f"{base_name}_scripted.pt")
        return original, 2
    
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version, reverse=True)
    latest_file = files[0]
    latest_version = extract_version(latest_file)
    return latest_file, latest_version + 1

ORIGINAL_MODEL, NEXT_VERSION = get_latest_model_and_next_version("emotion_model")
BUFFER_DIR = os.path.join(BASE_DIR, "buffers", "emotion")
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, "datasets", "fer2013", "train") 

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # Extract class from parent directory name
        class_name = os.path.basename(os.path.dirname(path))
        label = EMOTION_CLASSES.index(class_name)
        
        try:
            img = Image.open(path).convert('RGB')
        except:
            # Fallback for cv2 saved buffer images
            img_bgr = cv2.imread(path)
            img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    # Load Patient Buffer Data
    buffer_files = glob.glob(os.path.join(BUFFER_DIR, "*", "*.jpg"))
    if len(buffer_files) == 0:
        print("[ERR] No buffer data found! Cannot fine-tune.")
        return None, None

    buffer_dataset = EmotionDataset(buffer_files, transform=transform)
    
    # Load Original Generic Data
    original_files = glob.glob(os.path.join(ORIGINAL_DATA_DIR, "*", "*.jpg"))
    if len(original_files) == 0:
        print("[WARN] No original training data found in fer2013/train! Falling back to buffer only.")
        combined_dataset = buffer_dataset
    else:
        # Calculate how many original samples to randomly pick for the 85/15 ratio
        # Let B = len(buffer) which is 15%. Total T = B / 0.15
        # Original amount needed = T - B = (B / 0.15) - B
        num_buffer = len(buffer_dataset)
        num_original_needed = int((num_buffer / NEW_DATA_RATIO) - num_buffer)
        
        if num_original_needed > len(original_files):
            num_original_needed = len(original_files)
            
        selected_original_files = random.sample(original_files, num_original_needed)
        original_dataset = EmotionDataset(selected_original_files, transform=transform)
        print(f"[INFO] Mixing {num_buffer} patient frames with {num_original_needed} generic frames.")
        
        combined_dataset = ConcatDataset([buffer_dataset, original_dataset])

    loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    return loader, len(combined_dataset)

def retrain_emotion_model():
    print(f"--- Starting Emotion Continual Leaning Fine-Tuning ---")
    loader, total_samples = get_dataloaders()
    if loader is None: return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load Original Model
    try:
        model = torch.load(ORIGINAL_MODEL, map_location=device)
        if isinstance(model, dict):
            # Probably state dict, need architecture! 
            # If using timm it's a hassle without arch script. For now assuming full model was saved.
            print("[WARN] original model might be a state_dict, not full structure.")
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        return

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs) # might need extraction logic depending on model arch
            
            # Simple assumption of raw logits, if tuple returned, adjust
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(loader):.4f} | Acc: {epoch_acc:.2f}%")

    # Save Updated Models
    save_path_pth = os.path.join(BASE_DIR, f"emotion_model_full_v{NEXT_VERSION}.pth")
    torch.save(model.state_dict(), save_path_pth)
    print(f"✓ Saved v{NEXT_VERSION} state_dict to: {save_path_pth}")

    # TorchScript for real-time inference
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    try:
        scripted = torch.jit.trace(model, dummy_input)
        save_path_pt = os.path.join(BASE_DIR, f"emotion_model_full_v{NEXT_VERSION}.pt")
        scripted.save(save_path_pt)
        print(f"✓ Saved v{NEXT_VERSION} TorchScript to: {save_path_pt}")
    except Exception as e:
        print(f"[WARN] Failed to TorchScript trace. {e}")

    print(f"--- Fine-Tuning v{NEXT_VERSION} Complete ---")

if __name__ == "__main__":
    retrain_emotion_model()
