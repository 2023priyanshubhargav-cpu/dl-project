import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import glob
import numpy as np
import wave

# ==============================================================
# PHASE 2 - V2 EMBEDDING EXTRACTION
# Description: This script runs the newly trained v2 models over
# the buffer data to extract fresh embeddings for fusion retraining.
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUFFER_ROOT = os.path.join(BASE_DIR, "buffers")
OUT_DIR = os.path.join(BASE_DIR, "v2_embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_model(base_name, extension, fallback_default):
    import glob
    import re
    # Try to find highest _vX.extension
    pattern = os.path.join(BASE_DIR, f"{base_name}_v*{extension}")
    files = glob.glob(pattern)
    if not files:
        # Check for original with exact extension
        fallback = os.path.join(BASE_DIR, f"{base_name}{extension}")
        if os.path.exists(fallback): return fallback
        return fallback_default
    
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version, reverse=True)
    return files[0]

# Modality Configs (Strict Formats)
MODALITIES = {
    "emotion": {
        "model_path": get_latest_model("emotion_model_full", ".pth", os.path.join(BASE_DIR, "emotion_model_full.pth")),
        "classes": ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        "input_type": "image"
    },
    "gesture": {
        "model_path": get_latest_model("gesture_model", ".pt", os.path.join(BASE_DIR, "gesture_model_new.pt")),
        "classes": ['help', 'stop', 'yes', 'no', 'calm', 'attention', 'emergency', 'suspicious', 'cancel', 'unknown'],
        "input_type": "image"
    },
    "environment": {
        "model_path": get_latest_model("environment_model_full", ".pt", os.path.join(BASE_DIR, "environment_model_scripted.pt")),
        "classes": None, # Dynamic from buffer
        "input_type": "image"
    },
    "speech": {
        "model_path": get_latest_model("speech_model", ".pth", os.path.join(BASE_DIR, "speech_model_new.pth")),
        "classes": ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off'],
        "input_type": "audio"
    }
}

# Image Preprocessing
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_audio(path):
    with wave.open(path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio).unsqueeze(0) # [1, T]

def extract_modality_embeddings(name, config):
    print(f"--- Extracting {name.upper()} v2 embeddings ---")
    
    path = config['model_path']
    if not os.path.exists(path):
        print(f"[SKIP] Model not found: {path}")
        return
        
    try:
        if path.endswith(".pt"):
            model = torch.jit.load(path, map_location=DEVICE)
        else:
            # Handle .pth (Pickle)
            if name == "speech":
                # Special handling for SpeechModel class
                from train_speech_finetune import SpeechModel
                model = SpeechModel(num_classes=10).to(DEVICE)
                ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            elif name == "emotion":
                # Assuming EmotionModel is available or we use torch.load
                model = torch.load(path, map_location=DEVICE, weights_only=False)
            else:
                model = torch.load(path, map_location=DEVICE, weights_only=False)
        model.eval()
    except Exception as e:
        print(f"[ERR] Failed to load {name} model ({path}): {e}")
        return

    buffer_path = os.path.join(BUFFER_ROOT, name)
    if not os.path.exists(buffer_path):
        print(f"[SKIP] Buffer path not found: {buffer_path}")
        return

    all_embs = []
    all_labels = []
    
    # Get class list (use config or scan folders for environment)
    if config['classes']:
        classes = config['classes']
    else:
        classes = sorted([d for d in os.listdir(buffer_path) if os.path.isdir(os.path.join(buffer_path, d))])

    for cls_name in classes:
        cls_dir = os.path.join(buffer_path, cls_name)
        if not os.path.exists(cls_dir): continue
        
        label_idx = classes.index(cls_name)
        files = glob.glob(os.path.join(cls_dir, "*.*"))
        files = [f for f in files if not f.endswith(".json")] # Skip metadata
        
        print(f"  Processing {cls_name}: {len(files)} files")
        
        for f in files:
            try:
                with torch.no_grad():
                    if config['input_type'] == "image":
                        img = Image.open(f).convert('RGB')
                        x = img_transform(img).unsqueeze(0).to(DEVICE)
                        # Expecting model to have get_embedding or similar. 
                        # If TorchScript, we might need to call specific method if we traced it.
                        # Assuming the v2 model returns (embedding, logits) or just embedding in a specific output.
                        # For the v2 TorchScripts we saved, usually it's the raw forward pass.
                        # If forward returns logits, we might need to update the model to export get_embedding.
                        # However, for fusion fine-tuning, the plan says we use the embeddings.
                        
                        # TRICK: Most of our models are scripted with a 'get_embedding' method if available,
                        # or we can use the penultimate layer if we can access it.
                        # If it's a raw TorchScript, we'll try to call get_embedding()
                        if hasattr(model, "get_embedding"):
                            emb = model.get_embedding(x)
                        else:
                            # Fallback: run forward and hope it's embeddings, or we need to fix script.
                            # For our v2 scripts, let's assume we want the embeddings.
                            emb = model(x) 
                            if isinstance(emb, (list, tuple)): emb = emb[0]
                    else:
                        audio = load_audio(f).to(DEVICE)
                        if hasattr(model, "get_embedding"):
                            emb = model.get_embedding(audio)
                        else:
                            emb = model(audio)
                            if isinstance(emb, (list, tuple)): emb = emb[0]
                    
                    all_embs.append(emb.cpu().squeeze())
                    all_labels.append(label_idx)
            except Exception as e:
                pass

    if all_embs:
        stacked_embs = torch.stack(all_embs)
        save_data = {
            "embeddings": stacked_embs,
            "labels": all_labels,
            "classes": classes
        }
        out_path = os.path.join(OUT_DIR, f"{name}_v2_pool.pt")
        torch.save(save_data, out_path)
        print(f"✓ Saved {len(all_embs)} embeddings to {out_path}")
    else:
        print(f"[WARN] No embeddings extracted for {name}")

def main():
    for name, config in MODALITIES.items():
        extract_modality_embeddings(name, config)

if __name__ == "__main__":
    main()
