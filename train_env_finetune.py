import os
import glob
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NEW_DATA_RATIO = 0.15
ENV_TARGET_CLASSES = 5
def get_latest_model_and_next_version(base_name):
    import glob
    import re
    pattern = os.path.join(BASE_DIR, f"{base_name}_v*.pt")
    files = glob.glob(pattern)
    if not files:
        original = os.path.join(BASE_DIR, "environment_model_full.pth")
        if not os.path.exists(original):
            original = os.path.join(BASE_DIR, "environment_model_scripted.pt")
        return original, 2
    
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version, reverse=True)
    latest_file = files[0]
    latest_version = extract_version(latest_file)
    return latest_file, latest_version + 1

ORIGINAL_MODEL, NEXT_VERSION = get_latest_model_and_next_version("environment_model_full")
BUFFER_DIR = os.path.join(BASE_DIR, "buffers", "environment")
ORIGINAL_DATA_DIRS = [
    os.path.join(BASE_DIR, "datasets", "places365_filtered", "train"),
    os.path.join(BASE_DIR, "datasets", "places365", "train"),
]

ENV_CLASSES = [
    'alcove', 'alley', 'apartment_building-outdoor', 'archive', 'atrium-public', 
    'attic', 'auditorium', 'bakery-shop', 'balcony-exterior', 'balcony-interior', 
    'basement', 'bathroom', 'beauty_salon', 'bedchamber', 'bedroom', 
    'biology_laboratory', 'bookstore', 'bow_window-indoor', 'building_facade', 
    'bus_interior', 'bus_station-indoor', 'butchers_shop', 'cafeteria', 
    'campus', 'candy_store', 'car_interior', 'chemistry_lab', 'childs_room', 
    'church-indoor', 'classroom', 'clean_room', 'closet', 'clothing_store', 
    'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 
    'construction_site', 'corridor', 'courthouse', 'courtyard', 'crosswalk', 
    'delicatessen', 'department_store', 'dining_hall', 'dining_room', 
    'doorway-outdoor', 'dorm_room', 'downtown', 'dressing_room', 'driveway', 
    'drugstore', 'elevator-door', 'elevator_lobby', 'elevator_shaft', 
    'entrance_hall', 'escalator-indoor', 'fabric_store', 'fastfood_restaurant', 
    'fire_escape', 'fire_station', 'florist_shop-indoor', 'food_court', 
    'garage-indoor', 'garage-outdoor', 'gas_station', 'general_store-indoor', 
    'gift_shop', 'gymnasium-indoor', 'hardware_store', 'home_office', 
    'home_theater', 'hospital', 'hospital_room', 'hotel_room', 'house', 
    'ice_cream_parlor', 'jacuzzi-indoor', 'jewelry_shop', 'kindergarden_classroom', 
    'kitchen', 'laundromat', 'lawn', 'lecture_room', 'library-indoor', 
    'living_room', 'lobby', 'locker_room', 'mansion', 'market-indoor', 
    'martial_arts_gym', 'mezzanine', 'museum-indoor', 'natural_history_museum', 
    'nursery', 'nursing_home', 'office', 'office_building', 'office_cubicles', 
    'operating_room', 'pantry', 'park', 'parking_garage-indoor', 
    'parking_garage-outdoor', 'parking_lot', 'patio', 'pet_shop', 'pharmacy', 
    'physics_laboratory', 'picnic_area', 'pizzeria', 'playground', 'playroom', 
    'plaza', 'porch', 'promenade', 'reception', 'recreation_room', 
    'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'roof_garden', 
    'sauna', 'schoolhouse', 'science_museum', 'server_room', 'shoe_shop', 
    'shopfront', 'shopping_mall-indoor', 'shower', 'skyscraper', 'staircase', 
    'storage_room', 'street', 'subway_station-platform', 'supermarket', 
    'swimming_pool-indoor', 'television_room', 'ticket_booth', 'toyshop', 
    'train_interior', 'train_station-platform', 'utility_room', 
    'veterinarians_office', 'waiting_room', 'wet_bar', 'yard'
]
NUM_CLASSES = len(ENV_CLASSES)


class EnvironmentModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embedding_dim=512):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(1024, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.GELU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def get_embedding(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.embedding_head(x)

    def forward(self, x):
        return self.classifier(self.get_embedding(x))


def top_environment_classes(buffer_dir):
    counts = Counter()
    if not os.path.exists(buffer_dir):
        return []
    for class_name in os.listdir(buffer_dir):
        if class_name not in ENV_CLASSES:
            continue
        class_dir = os.path.join(buffer_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if count > 0:
            counts[class_name] = count
    return counts.most_common(ENV_TARGET_CLASSES)


class EnvironmentBufferDataset(Dataset):
    def __init__(self, buffer_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for class_name in ENV_CLASSES:
            class_dir = os.path.join(buffer_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            label = ENV_CLASSES.index(class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, f))
                    self.labels.append(label)

        if not self.images:
            raise RuntimeError("No real environment buffer images found. Refusing to train on dummy data.")

        num_buffer = len(self.images)
        num_original_needed = max(0, int((num_buffer / NEW_DATA_RATIO) - num_buffer))
        original_candidates = []
        
        # Pull original generic data for replay
        for class_name in ENV_CLASSES:
            label = ENV_CLASSES.index(class_name)
            for root in ORIGINAL_DATA_DIRS:
                class_dir = os.path.join(root, class_name)
                if not os.path.exists(class_dir):
                    continue
                original_candidates.extend(
                    (path, label)
                    for path in glob.glob(os.path.join(class_dir, "*"))
                    if path.lower().endswith(('.jpg', '.jpeg', '.png'))
                )

        if original_candidates:
            random.shuffle(original_candidates)
            selected_original = original_candidates[:min(num_original_needed, len(original_candidates))]
            self.images.extend(path for path, _ in selected_original)
            self.labels.extend(label for _, label in selected_original)
            print(
                f"[INFO] Environment replay mix: {num_buffer} patient/buffer images + "
                f"{len(selected_original)} original generic images."
            )
        else:
            print("[WARN] No matching original environment replay images found for top-5 classes.")

        print("[INFO] Environment top-5 classes:", ", ".join(c for c, _ in top_classes))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_environment_model(device):
    if not os.path.exists(ORIGINAL_MODEL):
        raise FileNotFoundError(f"Missing environment checkpoint: {ORIGINAL_MODEL}")
    model = torch.load(ORIGINAL_MODEL, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise RuntimeError(f"{ORIGINAL_MODEL} did not load as a trainable torch module.")
    return model


def retrain_environment_model():
    print("--- Starting ENVIRONMENT Continual Fine-Tuning ---")
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = EnvironmentBufferDataset(BUFFER_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"[INFO] Loaded {len(dataset)} real environment buffer images.")

    model = load_environment_model(device).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[1]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        acc = 100.0 * correct / max(1, total)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {running_loss / max(1, len(loader)):.4f} | Acc: {acc:.2f}%")

    save_path_pth = os.path.join(BASE_DIR, f"environment_model_full_v{NEXT_VERSION}.pth")
    torch.save(model.state_dict(), save_path_pth)
    print(f"Saved v{NEXT_VERSION} state_dict to: {save_path_pth}")

    model.eval()
    trace_input = torch.randn(1, 3, 224, 224).to(device)
    try:
        scripted = torch.jit.trace(model, trace_input)
        save_path_pt = os.path.join(BASE_DIR, f"environment_model_full_v{NEXT_VERSION}.pt")
        scripted.save(save_path_pt)
        print(f"Saved v{NEXT_VERSION} TorchScript to: {save_path_pt}")
    except Exception as e:
        print(f"[WARN] Trace failed: {e}")

    print(f"--- Environment Fine-Tuning v{NEXT_VERSION} Complete ---")


if __name__ == '__main__':
    retrain_environment_model()
