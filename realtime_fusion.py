# ==============================================================
# REAL-TIME MULTIMODAL FUSION — FINAL VERSION (WITH XAI LABELS)
# All 5 branch models + fusion via TorchScript (.pt)
# Speech via .pth + class def (Wav2Vec2 TorchScript incompatible)
# Modality dropout: any missing model auto-masked, fusion still runs
# Press Q to quit
# ==============================================================

import os, sys, time, queue, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import sounddevice as sd
from transformers import Wav2Vec2Model

warnings.filterwarnings("ignore")

# ==============================================================
# ▼▼▼  PATHS — all files in ~/emotion_project  ▼▼▼
# ==============================================================
BASE               = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE, "emotion_model_scripted.pt")
ENV_MODEL_PATH     = os.path.join(BASE, "environment_model_scripted.pt")
HEALTH_MODEL_PATH  = os.path.join(BASE, "health_model_scripted.pt")
GESTURE_MODEL_PATH = os.path.join(BASE, "gesture_model.pt")
SPEECH_MODEL_PATH  = os.path.join(BASE, "speech_model.pth")
FUSION_MODEL_PATH  = os.path.join(BASE, "best_fusion_model.pt")

CAMERA_INDEX      = 0
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHUNK_SECS  = 2
SPEECH_MODEL_SAMPLE_RATE = 16000
SPEECH_MODEL_SAMPLES     = 16000
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==============================================================
# SECTION 1 — CONFIG & CLASSES
# ==============================================================
MODALITY_DIMS   = [768, 512, 512, 512, 512] 
MODALITY_NAMES  = ['emotion', 'env', 'health', 'gesture', 'speech']
COMMON_DIM      = 256
NUM_CLASSES     = 4
FUSION_CLASSES  = ['normal', 'needs_attention', 'call_nurse', 'emergency']

# --- YOUR EXACT TRAINING CLASSES ---
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

HEALTH_CLASSES  = ['baseline', 'stress', 'amusement', 'meditation']

GESTURE_CLASSES = ['help', 'stop', 'yes', 'no', 'calm', 'attention', 'emergency', 'suspicious', 'cancel', 'unknown']

SPEECH_COMMANDS = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']

ENV_CLASSES = [
    'bathroom', 'bedroom', 'childs_room', 'clean_room', 'corridor', 'hospital', 'hospital_room', 
    'kitchen', 'living_room', 'nursery', 'nursing_home', 'operating_room', 'pharmacy', 'reception', 
    'shower', 'staircase', 'utility_room', 'veterinarians_office', 'waiting_room', 'alcove', 'attic', 
    'balcony-interior', 'basement', 'bedchamber', 'bow_window-indoor', 'closet', 'dining_room', 
    'dorm_room', 'dressing_room', 'home_office', 'home_theater', 'hotel_room', 'jacuzzi-indoor', 
    'locker_room', 'pantry', 'patio', 'playroom', 'porch', 'recreation_room', 'sauna', 'storage_room', 
    'television_room', 'wet_bar', 'yard', 'atrium-public', 'elevator-door', 'elevator_lobby', 
    'elevator_shaft', 'entrance_hall', 'escalator-indoor', 'fire_escape', 'garage-indoor', 
    'garage-outdoor', 'lawn', 'lobby', 'mezzanine', 'parking_garage-indoor', 'parking_garage-outdoor', 
    'parking_lot', 'roof_garden', 'street', 'archive', 'auditorium', 'biology_laboratory', 'bookstore', 
    'cafeteria', 'chemistry_lab', 'church-indoor', 'classroom', 'computer_room', 'conference_center', 
    'conference_room', 'courthouse', 'dining_hall', 'drugstore', 'fire_station', 'gymnasium-indoor', 
    'kindergarden_classroom', 'lecture_room', 'library-indoor', 'martial_arts_gym', 'museum-indoor', 
    'natural_history_museum', 'office', 'office_building', 'office_cubicles', 'physics_laboratory', 
    'schoolhouse', 'science_museum', 'server_room', 'shopping_mall-indoor', 'subway_station-platform', 
    'swimming_pool-indoor', 'bakery-shop', 'beauty_salon', 'butchers_shop', 'candy_store', 'clothing_store', 
    'coffee_shop', 'delicatessen', 'department_store', 'fabric_store', 'fastfood_restaurant', 
    'florist_shop-indoor', 'food_court', 'general_store-indoor', 'gift_shop', 'hardware_store', 
    'ice_cream_parlor', 'jewelry_shop', 'laundromat', 'market-indoor', 'pet_shop', 'pizzeria', 'restaurant', 
    'restaurant_kitchen', 'shoe_shop', 'shopfront', 'supermarket', 'toyshop', 'bus_interior', 
    'bus_station-indoor', 'car_interior', 'gas_station', 'train_interior', 'train_station-platform', 
    'alley', 'apartment_building-outdoor', 'balcony-exterior', 'building_facade', 'campus', 
    'construction_site', 'courtyard', 'crosswalk', 'doorway-outdoor', 'downtown', 'driveway', 'house', 
    'mansion', 'park', 'picnic_area', 'playground', 'plaza', 'promenade', 'residential_neighborhood', 
    'skyscraper', 'ticket_booth'
]

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

ACTION_DISPLAY = {
    'normal':          ('NORMAL',          (0, 200, 0)),
    'needs_attention': ('NEEDS ATTENTION', (0, 165, 255)),
    'call_nurse':      ('CALL NURSE',      (0, 0, 255)),
    'emergency':       ('!! EMERGENCY !!', (0, 0, 180)),
}

# ==============================================================
# SECTION 2 — SPEECH CLASS DEF
# ==============================================================
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
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features  = self.wav2vec(x).last_hidden_state.mean(dim=1)
        embedding = self.embedding(features)
        return embedding, self.classifier(embedding)

    def get_embedding(self, x):
        emb, _ = self.forward(x)
        return emb

# ==============================================================
# SECTION 3 — LOAD ALL MODELS
# ==============================================================
def load_jit(path, name):
    if not os.path.exists(path):
        print(f"  [{name}] NOT FOUND → will be masked")
        return None
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        print(f"  [{name}] ✅")
        return m
    except Exception as e:
        print(f"  [{name}] FAILED: {e} → will be masked")
        return None

print("\nLoading branch models...")
emotion_model = load_jit(EMOTION_MODEL_PATH,  'emotion')
env_model     = load_jit(ENV_MODEL_PATH,      'environment')
health_model  = load_jit(HEALTH_MODEL_PATH,   'health')
gesture_model = load_jit(GESTURE_MODEL_PATH,  'gesture')

speech_model = None
if os.path.exists(SPEECH_MODEL_PATH):
    try:
        speech_model = SpeechModel(num_classes=len(SPEECH_COMMANDS)).to(device)
        ckpt = torch.load(SPEECH_MODEL_PATH, map_location=device, weights_only=False)
        speech_model.load_state_dict(ckpt['model_state_dict'])
        speech_model.eval()
        print(f"  [speech] ✅")
    except Exception as e:
        print(f"  [speech] FAILED: {e} → will be masked")
        speech_model = None
else:
    print(f"  [speech] NOT FOUND → will be masked")

print("\nLoading fusion model...")
if not os.path.exists(FUSION_MODEL_PATH):
    print(f"FATAL: fusion model not found at {FUSION_MODEL_PATH}")
    sys.exit(1)
try:
    fusion_model = torch.jit.load(FUSION_MODEL_PATH, map_location=device)
    fusion_model.eval()
    print("  [fusion] ✅")
except Exception as e:
    print(f"  [fusion] FAILED: {e}")
    sys.exit(1)

active = [n for n, m in zip(
    MODALITY_NAMES,
    [emotion_model, env_model, health_model, gesture_model, speech_model]
) if m is not None]
print(f"\nActive modalities: {active}")
print(f"Masked modalities: {[n for n in MODALITY_NAMES if n not in active]}")
print("Fusion will auto-mask any missing modality at runtime.\n")

# ==============================================================
# SECTION 4 — TRANSFORMS
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
# SECTION 5 — EMBEDDING EXTRACTION (NOW RETURNS LABELS TOO)
# ==============================================================
def smart_extract(out, expected_dim):
    emb = None
    if isinstance(out, tuple):
        for t in out:
            if isinstance(t, torch.Tensor) and len(t.shape) >= 2 and t.shape[-1] == expected_dim:
                emb = t
                break
        if emb is None: emb = out[0]
    else:
        emb = out

    if emb.shape[-1] != expected_dim:
        padded = torch.zeros(1, expected_dim, device=device)
        size = min(emb.shape[-1], expected_dim)
        padded[0, :size] = emb[0, :size]
        emb = padded
    return emb

def get_class_label(raw_out, class_list):
    """Safely extracts the predicted class name from the raw model output"""
    try:
        logits = raw_out[0] if isinstance(raw_out, tuple) else raw_out
        idx = logits.argmax(-1).item()
        return class_list[idx % len(class_list)]
    except:
        return "unk"

@torch.no_grad()
def get_emotion_embedding(frame_bgr):
    if emotion_model is None: return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x   = emotion_transform(img).unsqueeze(0).to(device)
        out = emotion_model(x)
        
        label = get_class_label(out, EMOTION_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[0])
        return F.normalize(emb, p=2, dim=1), label
    except Exception as e:
        print(f"[ERROR] Emotion model crashed: {e}")
        return None, None

@torch.no_grad()
def get_env_embedding(frame_bgr):
    if env_model is None: return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x   = env_transform(img).unsqueeze(0).to(device)
        out = env_model(x)
        
        label = get_class_label(out, ENV_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[1])
        return F.normalize(emb, p=2, dim=1), label
    except Exception as e:
        print(f"[ERROR] Env model crashed: {e}")
        return None, None

@torch.no_grad()
def get_health_embedding(signal_tensor=None):
    if health_model is None or signal_tensor is None: return None, None
    try:
        out = health_model(signal_tensor.to(device))
        label = get_class_label(out, HEALTH_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[2])
        return F.normalize(emb, p=2, dim=1), label
    except Exception as e:
        print(f"[ERROR] Health model crashed: {e}")
        return None, None

@torch.no_grad()
def get_gesture_embedding(frame_bgr):
    if gesture_model is None: return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x   = gesture_transform(img).unsqueeze(0).to(device)
        out = gesture_model(x)
        
        label = get_class_label(out, GESTURE_CLASSES)
        emb = smart_extract(out, MODALITY_DIMS[3])
        return F.normalize(emb, p=2, dim=1), label
    except Exception as e:
        print(f"[ERROR] Gesture model crashed: {e}")
        return None, None

@torch.no_grad()
def get_speech_embedding(audio_chunk):
    if speech_model is None or audio_chunk is None: return None, None
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
            
        out = speech_model.get_embedding(waveform)
        label = get_class_label(out, SPEECH_COMMANDS)
        emb = smart_extract(out, MODALITY_DIMS[4])
        return F.normalize(emb, p=2, dim=1), label
    except Exception as e:
        print(f"[ERROR] Speech model crashed: {e}")
        return None, None

# ==============================================================
# SECTION 6 — AUDIO THREAD
# ==============================================================
audio_queue   = queue.Queue(maxsize=2)
AUDIO_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECS)

def audio_callback(indata, frames, time_info, status):
    chunk = indata[:, 0].copy().astype(np.float32)
    if not audio_queue.full():
        audio_queue.put(chunk)

def start_audio():
    try:
        stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE, channels=1,
            dtype='float32', blocksize=AUDIO_SAMPLES,
            callback=audio_callback,
        )
        stream.start()
        print("Mic ✅")
        return stream
    except Exception as e:
        print(f"Mic unavailable ({e}) → speech masked")
        return None

# ==============================================================
# SECTION 7 — FUSION INFERENCE
# ==============================================================
@torch.no_grad()
def run_fusion(emotion_emb, env_emb, health_emb, gesture_emb, speech_emb):
    fusion_model.eval()

    raw  = [emotion_emb, env_emb, health_emb, gesture_emb, speech_emb]
    embs = []
    mask = []

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

    attn = {MODALITY_NAMES[i]: round(attn_w[0, i].item(), 3) for i in range(5)}
    used = [MODALITY_NAMES[i] for i in range(5) if mask[i] == 1.0]

    return FUSION_CLASSES[pred], round(probs[pred].item()*100, 1), probs.tolist(), attn, used

# ==============================================================
# SECTION 8 — DISPLAY OVERLAY (UPDATED FOR LABELS)
# ==============================================================
def draw_overlay(frame, action, conf, probs, attn, used, fps, labels):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, h), (0, 0, 0), -1)  # Made UI slightly wider to fit text
    frame   = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    label, color = ACTION_DISPLAY[action]
    cv2.rectangle(frame, (0, 0), (330, 60), color, -1)
    cv2.putText(frame, label,               (8, 42),  cv2.FONT_HERSHEY_DUPLEX,  1.0, (255,255,255), 2)
    cv2.putText(frame, f"Confidence: {conf:.1f}%", (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(220,220,220), 1)

    cv2.putText(frame, "Class probabilities:", (8, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170,170,170), 1)
    bar_cols = [(0,200,0),(0,165,255),(0,0,220),(0,0,160)]
    for i, (cls, p) in enumerate(zip(FUSION_CLASSES, probs)):
        y = 120 + i * 28
        cv2.rectangle(frame, (8, y), (8 + int(p*200), y+16), bar_cols[i], -1)
        cv2.rectangle(frame, (8, y), (208, y+16), (90,90,90), 1)
        cv2.putText(frame, f"{cls.replace('_',' ')}  {p*100:.0f}%",
                    (10, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255,255,255), 1)

    y_start = 120 + 4*28 + 12
    cv2.putText(frame, "Attention & Predictions:", (8, y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170,170,170), 1)
                
    for i, (mod, w) in enumerate(attn.items()):
        y   = y_start + 16 + i*24
        active_mod = mod in used
        col = (80,210,80) if active_mod else (60,60,60)
        cv2.rectangle(frame, (8, y), (8 + int(w*200), y+15), col, -1)
        cv2.rectangle(frame, (8, y), (208, y+15), (70,70,70), 1)
        
        # New Logic: Insert the class name if active
        if active_mod:
            pred_lbl = labels.get(mod, "unk")
            # Truncate long labels so UI stays clean
            if len(pred_lbl) > 13: pred_lbl = pred_lbl[:11] + ".."
            status_tag = f"({pred_lbl})"
        else:
            status_tag = "(--)"
            
        cv2.putText(frame, f"{mod} {status_tag}   {w:.3f}",
                    (10, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)

    y_bot = y_start + 16 + 5*24 + 10
    n_active = len(used)
    cv2.putText(frame, f"Active: {n_active}/5  [{', '.join(used) or 'none'}]",
                (8, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120,255,120), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1)

    return frame

# ==============================================================
# SECTION 9 — MAIN LOOP
# ==============================================================
def main():
    print("Starting real-time multimodal fusion...")
    print("Any model that failed to load is auto-masked — system still runs.")
    print("Press Q to quit.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Multimodal Fusion — Real-time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multimodal Fusion — Real-time", 640, 480)

    audio_stream = start_audio()

    last_action = 'normal'
    last_conf   = 0.0
    last_probs  = [0.25] * 4
    last_attn   = {n: 0.0 for n in MODALITY_NAMES}
    last_used   = []
    last_labels = {}
    last_audio  = None
    fps         = 0.0
    frame_count = 0
    fps_timer   = time.time()
    INFER_EVERY = 3   

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Waiting for valid camera frame...")
                time.sleep(0.1)
                continue

            frame_count += 1

            try:
                last_audio = audio_queue.get_nowait()
            except queue.Empty:
                pass

            if frame_count % INFER_EVERY == 0:
                emotion_emb, emotion_lbl = get_emotion_embedding(frame)
                env_emb, env_lbl         = get_env_embedding(frame)
                health_emb, health_lbl   = get_health_embedding(None)   
                gesture_emb, gesture_lbl = get_gesture_embedding(frame)
                speech_emb, speech_lbl   = get_speech_embedding(last_audio)

                # Store labels to send to UI
                last_labels = {
                    'emotion': emotion_lbl,
                    'env': env_lbl,
                    'health': health_lbl,
                    'gesture': gesture_lbl,
                    'speech': speech_lbl
                }

                last_action, last_conf, last_probs, last_attn, last_used = run_fusion(
                    emotion_emb, env_emb, health_emb, gesture_emb, speech_emb
                )

            if frame_count % 30 == 0:
                elapsed   = time.time() - fps_timer + 1e-6
                fps       = 30.0 / elapsed
                fps_timer = time.time()

            out = draw_overlay(
                frame.copy(), last_action, last_conf,
                last_probs, last_attn, last_used, fps, last_labels
            )
            cv2.imshow("Multimodal Fusion — Real-time", out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if audio_stream:
            audio_stream.stop()
            audio_stream.close()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()
