# ==============================================================
# REAL-TIME MULTIMODAL FUSION — 8-CLASS VERSION
# Classes: normal | needs_attention | call_nurse | emergency |
#          agitated | distressed_calm | sudden_shock | uncooperative
# Branch models: emotion | env | health | gesture | speech
# Fusion: TorchScript .pt  (best_fusion_model_8cls.pt)
# Missing model → auto-masked, fusion still runs
# Press Q to quit
# ==============================================================

import os, sys, time, queue, warnings
from buffer_manager import BufferManager
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import sounddevice as sd
from ip_audio_streamer import IPWebcamAudioStreamer
try:
    import timm
except ImportError:
    print("[WARN] timm not installed. Emotion .pth might fail to load.")
try:
    from transformers import Wav2Vec2Model
except Exception as e:
    print(f"[WARN] transformers import failed: {e}. Speech modality disabled.")
    Wav2Vec2Model = None

# Import RL smoother for stabilizing predictions
try:
    from ppo_inference import get_ppo_smoother
    RL_AVAILABLE = True
except ImportError:
    print("[WARN] RL inference module not available. Predictions will not be smoothed.")
    RL_AVAILABLE = False


warnings.filterwarnings("ignore")

# ==============================================================
# ▼▼▼  PATHS — edit only this block  ▼▼▼
# ==============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

# Dynamic Model Path Helper
def get_latest_model(base_name, extension, default_fallback):
    import glob
    import re
    # Look for files like base_name_v*.extension
    pattern = os.path.join(BASE, f"{base_name}_v*{extension}")
    files = glob.glob(pattern)
    
    if not files:
        fallback = os.path.join(BASE, f"{base_name}{extension}")
        if os.path.exists(fallback): return fallback
        return default_fallback
    
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version, reverse=True)
    return files[0]

# --- Dynamic Path Loading (Automatic v2, v3, v4...) ---
# Strict Formats: Emotion/Speech (.pth) | Env/Health/Gesture/Fusion (.pt)
EMOTION_MODEL_PATH  = get_latest_model("emotion_model_full", ".pth", os.path.join(BASE, "emotion_model_full.pth"))
ENV_MODEL_PATH      = get_latest_model("environment_model_full", ".pt",  os.path.join(BASE, "environment_model_scripted.pt"))
HEALTH_MODEL_PATH   = get_latest_model("health_model_scripted", ".pt",  os.path.join(BASE, "health_model_scripted.pt"))
GESTURE_MODEL_PATH  = get_latest_model("gesture_model", ".pth",         os.path.join(BASE, "gesture_model.pth"))
SPEECH_MODEL_PATH   = get_latest_model("speech_model", ".pth",         os.path.join(BASE, "speech_model_new.pth"))
FUSION_MODEL_PATH   = get_latest_model("best_fusion_model_8cls", ".pt", os.path.join(BASE, "best_fusion_model_8cls.pt"))

CAMERA_INDEX        = 0
AUDIO_SAMPLE_RATE   = 48000
AUDIO_CHUNK_SECS    = 1
SPEECH_MODEL_SAMPLE_RATE = 16000
SPEECH_MODEL_SAMPLES     = 16000

# ────────────────────────────────────────────────────────────
# IP WEBCAM AUDIO INPUT (Optional) — Set to use mobile phone audio
# ────────────────────────────────────────────────────────────
USE_IP_WEBCAM_AUDIO = True  # Set True to use IP Webcam, False for system mic
IP_WEBCAM_IP        = "192.168.2.1"  # ← Change this to your IP Webcam's IP
IP_WEBCAM_PORT      = 8080  # Default IP Webcam port
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ==============================================================
# SECTION 1 — CONFIG & 8-CLASS LABELS
# ==============================================================
MODALITY_DIMS   = [768, 512, 512, 512, 512]   # emotion | env | health | gesture | speech
MODALITY_NAMES  = ['emotion', 'env', 'health', 'gesture', 'speech']
NUM_MODALITIES  = 5
COMMON_DIM      = 256
NUM_CLASSES     = 8

FUSION_CLASSES = [
    'normal',           # 0
    'needs_attention',  # 1
    'call_nurse',       # 2
    'emergency',        # 3
    'agitated',         # 4
    'distressed_calm',  # 5
    'sudden_shock',     # 6
    'uncooperative',    # 7
]

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
HEALTH_CLASSES  = ['baseline', 'stress', 'amusement', 'meditation']
GESTURE_CLASSES = ['help', 'stop', 'yes', 'no', 'calm', 'attention',
                   'emergency', 'suspicious', 'cancel', 'unknown']
SPEECH_COMMANDS = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Colour + display text per class (BGR for OpenCV)
ACTION_DISPLAY = {
    'normal':          ('NORMAL',            (0, 200, 0)),
    'needs_attention': ('NEEDS ATTENTION',   (0, 165, 255)),
    'call_nurse':      ('CALL NURSE',        (0, 0, 255)),
    'emergency':       ('!! EMERGENCY !!',   (0, 0, 180)),
    'agitated':        ('AGITATED',          (0, 100, 255)),
    'distressed_calm': ('DISTRESSED CALM',   (180, 60, 200)),
    'sudden_shock':    ('SUDDEN SHOCK',      (0, 200, 255)),
    'uncooperative':   ('UNCOOPERATIVE',     (30, 30, 200)),
}

# Robot / nurse text responses
ACTION_RESPONSES = {
    'normal':          "Patient appears calm and stable. Continuing routine monitoring.",
    'needs_attention': "Patient may need a check-in. Approaching now.",
    'call_nurse':      "Alerting nurse — patient requires assistance.",
    'emergency':       "EMERGENCY — calling for immediate medical help!",
    'agitated':        "Patient is agitated. Alerting staff for de-escalation.",
    'distressed_calm': "Patient shows emotional distress without physical stress. Flagging for mental-health check-in.",
    'sudden_shock':    "Sudden change detected — checking on patient immediately.",
    'uncooperative':   "Patient appears to be refusing care. Notifying staff.",
}

# ==============================================================
# SECTION 2 — CUSTOM MODEL CLASSES
# ==============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        self.cbam = CBAM(768)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def get_embedding(self, x):
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
    def forward(self, x):
        return self.classifier(self.get_embedding(x))

class GestureEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features  = self.backbone(x)
        embedding = self.embedding(features)
        logits    = self.classifier(embedding)
        return embedding, logits


if Wav2Vec2Model is not None:
    class SpeechModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                'facebook/wav2vec2-base', use_safetensors=True
            )
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
            self.embedding = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(512, num_classes)

        def forward(self, x):
            features  = self.wav2vec(x).last_hidden_state.mean(dim=1)
            embedding = self.embedding(features)
            return embedding, self.classifier(embedding)

        def get_embedding(self, x):
            emb, _ = self.forward(x)
            return emb
else:
    SpeechModel = None

# ==============================================================
# SECTION 3 — MODEL LOADING
# ==============================================================
def load_jit(path, name):
    if not os.path.exists(path):
        print(f"  [{name}] NOT FOUND → masked")
        return None
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        print(f"  [{name}] ✅")
        return m
    except Exception as e:
        print(f"  [{name}] FAILED: {e} → masked")
        return None

print("\nLoading branch models...")
emotion_model = None
if os.path.exists(EMOTION_MODEL_PATH):
    try:
        try:
            # Try TorchScript first
            emotion_model = torch.jit.load(EMOTION_MODEL_PATH, map_location=device)
            print(f"  [emotion] ✅ (TorchScript v{EMOTION_MODEL_PATH.split('_v')[-1].split('.')[0] if '_v' in EMOTION_MODEL_PATH else '?'})")
        except Exception:
            # Fallback to standard pickle load
            emotion_model = torch.load(EMOTION_MODEL_PATH, map_location=device, weights_only=False)
            print("  [emotion] ✅ (Standard .pth/.pt)")
        emotion_model.eval()
    except Exception as e:
        print(f"  [emotion] FAILED: {e} → masked")
else:
    print("  [emotion] NOT FOUND → masked")

env_model     = load_jit(ENV_MODEL_PATH,     'environment')
health_model  = load_jit(HEALTH_MODEL_PATH,  'health')

# Gesture uses torch.load because it was Pickled, not JIT exported
gesture_model = None
if os.path.exists(GESTURE_MODEL_PATH):
    try:
        try:
            # Try TorchScript first
            gesture_model = torch.jit.load(GESTURE_MODEL_PATH, map_location=device)
            print(f"  [gesture] ✅ (TorchScript v{GESTURE_MODEL_PATH.split('_v')[-1].split('.')[0] if '_v' in GESTURE_MODEL_PATH else '?'})")
        except Exception:
            # Fallback to standard pickle load
            gesture_model = GestureEncoder(num_classes=10).to(device)
            ckpt = torch.load(GESTURE_MODEL_PATH, map_location=device, weights_only=False)
            gesture_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print("  [gesture] ✅ (Standard .pth/.pt)")
        gesture_model.eval()
    except Exception as e:
        print(f"  [gesture] FAILED: {e} → masked")
else:
    print("  [gesture] NOT FOUND → masked")

speech_model = None
if Wav2Vec2Model is not None and os.path.exists(SPEECH_MODEL_PATH):
    try:
        if SPEECH_MODEL_PATH.endswith(".pt"):
            speech_model = torch.jit.load(SPEECH_MODEL_PATH, map_location=device)
            print(f"  [speech] ✅ (TorchScript v{SPEECH_MODEL_PATH.split('_v')[-1].split('.')[0] if '_v' in SPEECH_MODEL_PATH else '?'})")
        else:
            speech_model = SpeechModel(num_classes=len(SPEECH_COMMANDS)).to(device)
            ckpt = torch.load(SPEECH_MODEL_PATH, map_location=device, weights_only=False)
            speech_model.load_state_dict(ckpt['model_state_dict'])
            print("  [speech] ✅ (Original .pth)")
        speech_model.eval()
    except Exception as e:
        print(f"  [speech] FAILED: {e} → masked")
        speech_model = None
else:
    print("  [speech] NOT FOUND → masked")

print("\nLoading 8-class fusion model...")
if not os.path.exists(FUSION_MODEL_PATH):
    print(f"FATAL: {FUSION_MODEL_PATH} not found")
    sys.exit(1)
try:
    fusion_model = torch.jit.load(FUSION_MODEL_PATH, map_location=device)
    fusion_model.eval()
    print("  [fusion-8cls] ✅")
except Exception as e:
    print(f"  [fusion-8cls] FAILED: {e}")
    sys.exit(1)

active_mods = [n for n, m in zip(
    MODALITY_NAMES,
    [emotion_model, env_model, health_model, gesture_model, speech_model]
) if m is not None]
masked_mods = [n for n in MODALITY_NAMES if n not in active_mods]
print(f"\nActive modalities : {active_mods}")
print(f"Masked modalities : {masked_mods}")
print("Fusion auto-masks any missing modality at runtime.\n")

# ==============================================================
# SECTION 4 — IMAGE TRANSFORMS
# ==============================================================
emotion_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

env_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

gesture_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ==============================================================
# SECTION 5 — EMBEDDING HELPERS
# ==============================================================
def smart_extract(out, expected_dim):
    """Pull the embedding tensor of shape [1, expected_dim] from model output."""
    emb = None
    if isinstance(out, tuple):
        for t in out:
            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[-1] == expected_dim:
                emb = t
                break
        if emb is None:
            emb = out[0]
    else:
        emb = out

    if emb.shape[-1] != expected_dim:
        padded = torch.zeros(1, expected_dim, device=device)
        size = min(emb.shape[-1], expected_dim)
        padded[0, :size] = emb[0, :size]
        emb = padded
    return emb

def get_class_label(raw_out, class_list):
    """Return (label, confidence_pct) from raw model output."""
    try:
        # If output is (embedding, logits), logits is usually the smaller one or matches class_list
        if isinstance(raw_out, tuple):
            logits = raw_out[1] if len(raw_out) > 1 else raw_out[0]
            for t in raw_out:
                if isinstance(t, torch.Tensor) and t.shape[-1] == len(class_list):
                    logits = t
                    break
        else:
            logits = raw_out
        
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        label = class_list[idx.item() % len(class_list)]
        return label, conf.item() * 100
    except Exception:
        return "unk", 0.0

# ==============================================================
# SECTION 6 — PER-MODALITY EMBEDDING EXTRACTION
# ==============================================================
@torch.no_grad()
def get_emotion_embedding(frame_bgr):
    if emotion_model is None:
        return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x   = emotion_transform(img).unsqueeze(0).to(device)
        
        if hasattr(emotion_model, 'get_embedding'):
            emb = emotion_model.get_embedding(x)
            logits = emotion_model.classifier(emb)
            out = (emb, logits)
        else:
            out = emotion_model(x)

        lbl_name, lbl_conf = get_class_label(out, EMOTION_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[0])
        return F.normalize(emb, p=2, dim=1), (lbl_name, lbl_conf)
    except Exception as e:
        print(f"[ERR] emotion: {e}")
        return None, None

@torch.no_grad()
def get_env_embedding(frame_bgr):
    if env_model is None:
        return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x   = env_transform(img).unsqueeze(0).to(device)
        out = env_model(x)
        lbl = get_class_label(out, ENV_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[1])
        return F.normalize(emb, p=2, dim=1), lbl
    except Exception as e:
        print(f"[ERR] env: {e}")
        return None, None

@torch.no_grad()
def get_health_embedding(signal_tensor=None):
    if health_model is None or signal_tensor is None:
        return None, None
    try:
        out = health_model(signal_tensor.to(device))
        lbl = get_class_label(out, HEALTH_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[2])
        return F.normalize(emb, p=2, dim=1), lbl
    except Exception as e:
        print(f"[ERR] health: {e}")
        return None, None

@torch.no_grad()
def get_gesture_embedding(frame_bgr):
    if gesture_model is None:
        return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x   = gesture_transform(img).unsqueeze(0).to(device)
        out = gesture_model(x)
        lbl_name, lbl_conf = get_class_label(out, GESTURE_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[3])
        return F.normalize(emb, p=2, dim=1), (lbl_name, lbl_conf)
    except Exception as e:
        print(f"[ERR] gesture: {e}")
        return None, None

@torch.no_grad()
def get_speech_embedding(audio_chunk):
    if speech_model is None or audio_chunk is None:
        return None, None
    try:
        audio_np = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

        if AUDIO_SAMPLE_RATE != SPEECH_MODEL_SAMPLE_RATE and audio_np.size > 0:
            duration = audio_np.size / float(AUDIO_SAMPLE_RATE)
            target_size = max(1, int(round(duration * SPEECH_MODEL_SAMPLE_RATE)))
            old_x = np.linspace(0.0, duration, num=audio_np.size, endpoint=False)
            new_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
            audio_np = np.interp(new_x, old_x, audio_np).astype(np.float32)

        waveform = torch.from_numpy(audio_np).unsqueeze(0).to(device)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        target = SPEECH_MODEL_SAMPLES
        if waveform.size(1) > target:
            waveform = waveform[:, :target]
        else:
            waveform = F.pad(waveform, (0, target - waveform.size(1)))
        # Use forward to get both embedding and logits for label/confidence
        emb, logits = speech_model(waveform)
        lbl = get_class_label((emb, logits), SPEECH_COMMANDS)
        return F.normalize(emb, p=2, dim=1), lbl
    except Exception as e:
        print(f"[ERR] speech: {e}")
        return None, None

# ==============================================================
# SECTION 7 — AUDIO THREAD
# ==============================================================
audio_queue   = queue.Queue(maxsize=2)
AUDIO_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECS)

def audio_callback(indata, frames, time_info, status):
    chunk = indata[:, 0].copy().astype(np.float32)
    if not audio_queue.full():
        audio_queue.put(chunk)

def start_audio():
    """Start audio input — either from system mic or IP Webcam."""
    try:
        if USE_IP_WEBCAM_AUDIO:
            # Use IP Webcam audio
            print(f"🎤 Connecting to IP Webcam audio: {IP_WEBCAM_IP}:{IP_WEBCAM_PORT}")
            streamer = IPWebcamAudioStreamer(
                ip_address=IP_WEBCAM_IP, 
                port=IP_WEBCAM_PORT,
                sample_rate=AUDIO_SAMPLE_RATE
            )
            streamer.start()
            print(f"✅ IP Webcam audio stream active")
            return streamer
        else:
            # Use system microphone (sounddevice)
            stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE, channels=1,
                dtype='float32', blocksize=AUDIO_SAMPLES,
                callback=audio_callback,
            )
            stream.start()
            print("🎤 System microphone ✅")
            return stream
    except Exception as e:
        print(f"🎤 Audio unavailable ({e}) → speech masked")
        return None

# ==============================================================
# SECTION 8 — FUSION INFERENCE (8 CLASSES)
# ==============================================================
@torch.no_grad()
def run_fusion(emotion_emb, env_emb, health_emb, gesture_emb, speech_emb):
    fusion_model.eval()
    raw  = [emotion_emb, env_emb, health_emb, gesture_emb, speech_emb]
    embs, mask = [], []

    for i, emb in enumerate(raw):
        if emb is not None:
            embs.append(emb.to(device))
            mask.append(1.0)
        else:
            embs.append(torch.zeros(1, MODALITY_DIMS[i], device=device))
            mask.append(0.0)

    mask_t         = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    logits, attn_w = fusion_model(*embs, mask_t)
    probs          = F.softmax(logits, dim=1).squeeze(0).cpu()
    pred           = probs.argmax().item()

    attn = {MODALITY_NAMES[i]: round(attn_w[0, i].item(), 3) for i in range(NUM_MODALITIES)}
    used = [MODALITY_NAMES[i] for i in range(NUM_MODALITIES) if mask[i] == 1.0]

    return FUSION_CLASSES[pred], round(probs[pred].item() * 100, 1), probs.tolist(), attn, used

# ==============================================================
# SECTION 9 — DRAW OVERLAY (8-CLASS UI)
# ==============================================================
# 8 bar colours (one per class)
BAR_COLORS = [
    (0, 200, 0),     # normal        — green
    (0, 165, 255),   # needs_att     — orange
    (0, 0, 255),     # call_nurse    — red
    (0, 0, 160),     # emergency     — dark red
    (0, 100, 255),   # agitated      — amber-red
    (180, 60, 200),  # distressed    — purple
    (0, 200, 255),   # sudden_shock  — yellow
    (30, 30, 200),   # uncooperative — deep red
]

def draw_overlay(frame, action, conf, probs, attn, used, fps, labels, progress):
    h, w    = frame.shape[:2]
    PANEL_W = 350
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, h), (0, 0, 0), -1)
    frame   = cv2.addWeighted(overlay, 0.50, frame, 0.50, 0)

    # ── Action header ─────────────────────────────────────────
    disp_text, color = ACTION_DISPLAY.get(action, (action.upper(), (120, 120, 120)))
    cv2.rectangle(frame, (0, 0), (PANEL_W, 58), color, -1)
    font_scale = 0.85 if len(disp_text) <= 14 else 0.65
    cv2.putText(frame, disp_text, (8, 40),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {conf:.1f}%", (8, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    # ── Robot response text (small, wraps) ────────────────────
    resp = ACTION_RESPONSES.get(action, "")
    # Truncate to fit panel
    max_chars = 48
    if len(resp) > max_chars:
        resp = resp[:max_chars - 2] + ".."
    cv2.putText(frame, resp, (8, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 230, 180), 1)

    # ── Top-3 class probability bars (model runs all 8 internally) ──
    y_title = 115
    cv2.putText(frame, "Top-3 predictions (of 8):", (8, y_title),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1)

    # Sort all 8 by probability descending, show only top 3
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
    for rank, (i, p) in enumerate(ranked):
        y     = y_title + 14 + rank * 26
        bar_w = int(p * (PANEL_W - 16))
        col   = BAR_COLORS[i]
        cv2.rectangle(frame, (8, y), (8 + bar_w, y + 18), col, -1)
        cv2.rectangle(frame, (8, y), (PANEL_W - 8, y + 18), (80, 80, 80), 1)
        short = FUSION_CLASSES[i].replace('_', ' ')
        star  = " \u2605" if FUSION_CLASSES[i] == action else ""   # star on winner
        cv2.putText(frame, f"{short}{star}  {p*100:.0f}%",
                    (11, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # ── Attention weights + per-modality prediction labels ────
    y_att = y_title + 14 + 3 * 26 + 10
    cv2.putText(frame, "Attention & predictions:", (8, y_att),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1)

    for i, (mod, w) in enumerate(attn.items()):
        y          = y_att + 14 + i * 22
        is_active  = mod in used
        bar_col    = (80, 210, 80) if is_active else (50, 50, 50)
        bar_w      = int(w * (PANEL_W - 16))
        cv2.rectangle(frame, (8, y), (8 + bar_w, y + 14), bar_col, -1)
        cv2.rectangle(frame, (8, y), (PANEL_W - 8, y + 14), (70, 70, 70), 1)

        pred_lbl = labels.get(mod, "--")
        if pred_lbl and len(pred_lbl) > 14:
            pred_lbl = pred_lbl[:12] + ".."
        status_tag = f"({pred_lbl})" if is_active else "(masked)"
        cv2.putText(frame, f"{mod} {status_tag}  {w:.3f}",
                    (11, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

    # ── Active modalities footer ──────────────────────────────
    y_bot = y_att + 14 + NUM_MODALITIES * 22 + 8
    n_active = len(used)
    cv2.putText(frame,
                f"Active: {n_active}/{NUM_MODALITIES}  [{', '.join(used) or 'none'}]",
                (8, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 255, 120), 1)

    # ── CONTINUAL LEARNING DASHBOARD (NEW) ──────────────────
    y_cl = y_bot + 25
    cv2.putText(frame, "LEARNING PROGRESS:", (8, y_cl),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 200, 255), 1)
    
    for i, (mod, data) in enumerate(progress.items()):
        if mod == "fusion": continue
        y = y_cl + 16 + i * 18
        ver = data['version']
        curr = data['current']
        tot = data['total']
        stat = data['status']
        
        # Color based on status
        color = (255, 255, 255)
        if stat == 'running': color = (0, 255, 255) # Yellow
        if stat == 'completed': color = (0, 255, 0) # Green
        
        # Progress Bar
        bar_max_w = 120
        p_ratio = min(1.0, curr / max(1, tot))
        bar_w = int(p_ratio * bar_max_w)
        cv2.rectangle(frame, (100, y - 9), (100 + bar_max_w, y + 2), (60, 60, 60), 1)
        cv2.rectangle(frame, (100, y - 9), (100 + bar_w, y + 2), color, -1)
        
        cv2.putText(frame, f"{mod[:3].upper()} v{ver}: {curr}/{tot}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

    cv2.putText(frame, f"FPS: {fps:.1f} | Brain: v{progress.get('fusion', {}).get('version', 1)}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 130, 130), 1)

    # ── Model tag (top-right corner) ─────────────────────────
    tag = "8-CLASS FUSION"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, tag, (int(w - tw - 6), 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 200, 255), 1)

    return frame

# ==============================================================
# SECTION 10 — MAIN LOOP
# ==============================================================
def main():
    print("=" * 60)
    print("  REAL-TIME 8-CLASS MULTIMODAL FUSION")
    print("=" * 60)
    print("  Classes:", FUSION_CLASSES)
    print("  Press Q to quit.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WIN = "Multimodal Fusion 8-class — Real-time"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 980, 480)   # wider to show 350-px panel + 640-px camera

    audio_stream = start_audio()

    # State carried across frames
    last_action = 'normal'
    last_conf   = 0.0
    last_probs  = [1.0 / NUM_CLASSES] * NUM_CLASSES
    last_attn   = {n: 0.0 for n in MODALITY_NAMES}
    last_used   = []
    last_labels = {}
    last_audio  = None
    fps         = 0.0
    frame_count = 0
    fps_timer   = time.time()
    INFER_EVERY = 3   # run inference every N frames (keeps UI smooth)
    
    # Initialize PPO smoother if available
    smoother = None
    if RL_AVAILABLE:
        try:
            smoother = get_ppo_smoother(model_path="ppo_smoother", window_size=5, num_classes=8)
            print(f"[PPO Smoother] {smoother.get_status()}")
        except Exception as e:
            print(f"[WARN] Failed to initialize PPO smoother: {e}")
            smoother = None

    # Initialize Continual Learning Buffer Manager
    buffer_mgr = BufferManager(environment_classes=ENV_CLASSES)
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Waiting for valid camera frame...")
                time.sleep(0.05)
                continue

            frame_count += 1

            # Get latest audio chunk (supports both system mic and IP Webcam)
            try:
                if USE_IP_WEBCAM_AUDIO and isinstance(audio_stream, IPWebcamAudioStreamer):
                    # For IP Webcam: pull from streamer's queue (0.1s timeout for Wi-Fi lag)
                    last_audio = audio_stream.get_chunk(timeout=0.1)
                else:
                    # For system mic: pull from audio_queue (callback populates it)
                    last_audio = audio_queue.get_nowait()
            except (queue.Empty, Exception):
                last_audio = None

            # Run inference every INFER_EVERY frames
            if frame_count % INFER_EVERY == 0:
                emotion_emb, emotion_lbl = get_emotion_embedding(frame)
                env_emb,     env_lbl     = get_env_embedding(frame)
                health_emb,  health_lbl  = get_health_embedding(None)   # (Masked - No sensor connected)
                gesture_emb, gesture_lbl = get_gesture_embedding(frame)
                speech_emb,  speech_lbl  = get_speech_embedding(last_audio)

                # === Continual Learning Data Collection ===
                if emotion_lbl and emotion_lbl[0] != 'unk':
                    buffer_mgr.save_image_frame(frame, 'emotion', emotion_lbl[0], emotion_lbl[1])
                if env_lbl and env_lbl[0] != 'unk':
                    buffer_mgr.save_image_frame(frame, 'environment', env_lbl[0], env_lbl[1])
                if gesture_lbl and gesture_lbl[0] != 'unk':
                    buffer_mgr.save_image_frame(frame, 'gesture', gesture_lbl[0], gesture_lbl[1])
                if speech_lbl and speech_lbl[0] != 'unk' and last_audio is not None:
                    buffer_mgr.save_audio_sample(last_audio, AUDIO_SAMPLE_RATE, speech_lbl[0], speech_lbl[1])
                
                # Check if it's time for manual (Fear/Sad) capture (e.g. roughly every 100 frames)
                if frame_count % (INFER_EVERY * 30) == 0:
                    buffer_mgr.check_and_handle_manual_capture(
                        cap, emotion_model, device, emotion_transform, get_class_label)
                    
                # === Autonomous Retraining Trigger System ===
                # Non-blocking check to see if any modality buffers reached their quota
                buffer_mgr.check_and_trigger_retraining()
                
                # ========================================

                last_labels = {
                    'emotion': emotion_lbl,
                    'env':     env_lbl,
                    'health':  health_lbl,
                    'gesture': gesture_lbl,
                    'speech':  speech_lbl,
                }

                (last_action, last_conf,
                 last_probs, last_attn, last_used) = run_fusion(
                    emotion_emb, env_emb, health_emb, gesture_emb, speech_emb
                )
                
                # Apply RL smoother if available to stabilize flickering
                if smoother is not None:
                    smoothed_pred, smoothed_conf = smoother.update(last_probs)
                    last_action = FUSION_CLASSES[smoothed_pred]
                    last_conf = smoothed_conf * 100

                # Console log with individual model confidences
                mod_details = []
                for mod_name in last_used:
                    if mod_name in last_labels and last_labels[mod_name]:
                        lbl, cnf = last_labels[mod_name]
                        mod_details.append(f"{mod_name}: {lbl} ({cnf:.1f}%)")
                
                details_str = " | ".join(mod_details)
                print(f"[{frame_count:05d}] "
                      f"Action={last_action:<15s} "
                      f"Conf={last_conf:.1f}% | "
                      f"{details_str}")

            # FPS calculation
            if frame_count % 30 == 0:
                elapsed   = time.time() - fps_timer + 1e-9
                fps       = 30.0 / elapsed
                fps_timer = time.time()

            # Get learning status
            progress = buffer_mgr.get_progress_summary()

            out = draw_overlay(
                frame.copy(), last_action, last_conf,
                last_probs, last_attn, last_used, fps, last_labels, progress
            )
            cv2.imshow(WIN, out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuit signal received.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if audio_stream:
            audio_stream.stop()
            if hasattr(audio_stream, 'close'):  # sounddevice stream has close()
                audio_stream.close()
        cv2.destroyAllWindows()
        buffer_mgr.check_and_trigger_retraining()  # Final retraining check
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
