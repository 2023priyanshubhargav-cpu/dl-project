import os
import glob
import random
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEECH_CLASSES = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']
NUM_CLASSES = len(SPEECH_CLASSES)
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
MODEL_SAMPLE_RATE = 16000
TARGET_LENGTH = 16000

def get_latest_model_and_next_version(base_name):
    import glob
    import re
    pattern = os.path.join(BASE_DIR, f"{base_name}_v*.pt")
    files = glob.glob(pattern)
    if not files:
        original = os.path.join(BASE_DIR, f"{base_name}_new.pth")
        return original, 2
    
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version, reverse=True)
    latest_file = files[0]
    latest_version = extract_version(latest_file)
    return latest_file, latest_version + 1

ORIGINAL_MODEL, NEXT_VERSION = get_latest_model_and_next_version("speech_model")
BUFFER_DIR = os.path.join(BASE_DIR, "buffers", "speech")


class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        self.wav2vec = Wav2Vec2Model(Wav2Vec2Config())
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
        self.embedding = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        outputs = self.wav2vec(x)
        features = outputs.last_hidden_state.mean(dim=1)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return embedding, logits


def resample_np(audio, from_rate, to_rate):
    if from_rate == to_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.size / float(from_rate)
    target_size = max(1, int(round(duration * to_rate)))
    old_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def preprocess_audio(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if np.max(np.abs(audio)) > 1.5:
        audio = audio / 32768.0
    audio = resample_np(audio, sample_rate, MODEL_SAMPLE_RATE)

    waveform = torch.from_numpy(audio)
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    if waveform.numel() > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]
    else:
        waveform = F.pad(waveform, (0, TARGET_LENGTH - waveform.numel()))
    return waveform


def read_wav_mono(path):
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width in {path}: {sample_width} bytes")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


class SpeechBufferDataset(Dataset):
    def __init__(self, buffer_dir):
        self.samples = []
        for label_idx, class_name in enumerate(SPEECH_CLASSES):
            class_dir = os.path.join(buffer_dir, class_name)
            for path in glob.glob(os.path.join(class_dir, "*.wav")):
                self.samples.append((path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No real speech buffer .wav files found under {buffer_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        sample_rate, audio = read_wav_mono(path)
        return preprocess_audio(audio, sample_rate), label


def load_speech_model(device):
    if not os.path.exists(ORIGINAL_MODEL):
        raise FileNotFoundError(f"Missing speech checkpoint: {ORIGINAL_MODEL}")

    model = SpeechModel(NUM_CLASSES).to(device)
    checkpoint = torch.load(ORIGINAL_MODEL, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model


def retrain_speech_model():
    print("--- Starting SPEECH Continual Fine-Tuning ---")
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = SpeechBufferDataset(BUFFER_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"[INFO] Loaded {len(dataset)} real speech buffer samples.")

    model = load_speech_model(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        acc = 100.0 * correct / max(1, total)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {running_loss / max(1, len(loader)):.4f} | Acc: {acc:.2f}%")

    save_path = os.path.join(BASE_DIR, f"speech_model_v{NEXT_VERSION}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "commands": SPEECH_CLASSES,
        "source": "continual_learning_real_buffer",
    }, save_path)
    print(f"Saved speech v{NEXT_VERSION} checkpoint to: {save_path}")

    # TorchScript export for easier embedding extraction
    model.eval()
    try:
        # Dummy input for wav2vec (16kHz audio, 2 seconds = 32000 samples)
        dummy_input = torch.randn(1, 32000).to(device)
        scripted = torch.jit.trace(model, dummy_input)
        save_path_pt = os.path.join(BASE_DIR, f"speech_model_v{NEXT_VERSION}.pt")
        scripted.save(save_path_pt)
        print(f"✓ Saved v{NEXT_VERSION} TorchScript to: {save_path_pt}")
    except Exception as e:
        print(f"[WARN] Speech TorchScript export failed: {e}")

    print(f"--- Speech Fine-Tuning v{NEXT_VERSION} Complete ---")


if __name__ == '__main__':
    retrain_speech_model()
