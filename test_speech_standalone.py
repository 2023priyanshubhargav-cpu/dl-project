import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from transformers import Wav2Vec2Model
from ip_audio_streamer import IPWebcamAudioStreamer

# --- Config ---
IP_ADDRESS = "192.168.2.1"
PORT = 8080
SPEECH_COMMANDS = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']
MODEL_PATH = "speech_model_new.pth"
SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000
CHUNK_SECS = 1.0  # 1 second for faster response
SAMPLES_TARGET = int(MODEL_SAMPLE_RATE * 1.0) # 16000

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Model Definition ---
class SpeechModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', use_safetensors=True)
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
        self.embedding = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.wav2vec(x).last_hidden_state.mean(dim=1)
        emb = self.embedding(features)
        return emb, self.classifier(emb)

# --- Loader ---
print(f"Loading speech model from {MODEL_PATH}...")
model = SpeechModel(num_classes=len(SPEECH_COMMANDS)).to(device)
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Check if it's state_dict or full model
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    else:
        model = ckpt
    model.eval()
    print("✓ Model loaded successfully.")
else:
    print(f"✗ ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

def get_meter(value, length=20):
    """Simple ASCII volume meter."""
    filled = int(value * length)
    return "[" + "#" * filled + "-" * (length - filled) + "]"

def test_speech():
    print(f"\n🎤 Connecting to IP Webcam audio at {IP_ADDRESS}:{PORT}...")
    streamer = IPWebcamAudioStreamer(ip_address=IP_ADDRESS, port=PORT, sample_rate=SAMPLE_RATE)
    
    try:
        streamer.start()
        print("\n" + "="*50)
        print("  SPEECH MODALITY STANDALONE TEST")
        print("  Commands:", SPEECH_COMMANDS)
        print("  Press Ctrl+C to stop.")
        print("="*50 + "\n")
        
        while True:
            chunk = streamer.get_chunk(timeout=2.0)
            if chunk is None:
                print("Waiting for audio data...")
                continue
            
            # 1. Visualization
            rms = np.sqrt(np.mean(chunk**2))
            meter = get_meter(min(1.0, rms * 5)) # scale for visibility
            
            # 2. Pre-process for model
            audio_np = np.asarray(chunk, dtype=np.float32).reshape(-1)
            
            # Resample to 16k
            duration = audio_np.size / float(SAMPLE_RATE)
            target_size = int(duration * MODEL_SAMPLE_RATE)
            old_x = np.linspace(0.0, duration, num=audio_np.size, endpoint=False)
            new_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
            audio_np = np.interp(new_x, old_x, audio_np).astype(np.float32)
            
            # Normalize and pad/clip
            waveform = torch.from_numpy(audio_np).unsqueeze(0).to(device)
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            if waveform.size(1) > SAMPLES_TARGET:
                waveform = waveform[:, :SAMPLES_TARGET]
            else:
                waveform = F.pad(waveform, (0, SAMPLES_TARGET - waveform.size(1)))
            
            # 3. Inference
            with torch.no_grad():
                _, logits = model(waveform)
                probs = F.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)
                
                label = SPEECH_COMMANDS[pred.item()]
                confidence = conf.item() * 100
                
            # 4. Display
            status = f"Volume: {meter} | Command: {label:<8s} ({confidence:.1f}%)"
            # Clear line and print
            sys.stdout.write("\r" + status)
            sys.stdout.flush()
            
            if confidence > 80 and label != 'unk':
                print(f"\n   >> [DETECTED] '{label.upper()}' at {confidence:.1f}% confidence")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()

if __name__ == "__main__":
    test_speech()
