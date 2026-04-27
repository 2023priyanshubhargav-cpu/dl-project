# 🧠 Continual Deep learning for Adaptive perception and Decision making for Human-Robot Interaction

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg) ![Transformers](https://img.shields.io/badge/Transformers-Wav2Vec2-green.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📑 Project Essence
The **Continual Deep learning for Adaptive perception and Decision making for Human-Robot Interaction** project is an industrial-grade platform that integrates five distinct sensory modalities to continuously monitor patient status. By employing a late-fusion multimodal architecture, the system is highly robust against single-sensor failures. 

Furthermore, the system is designed with a **Self-Evolving Iterative Loop**: it detects inference uncertainty in real-time, autonomously saves the surrounding "edge-case" sensory data, and seamlessly integrates these cases into future model fine-tuning iterations.

---

## 🔬 Core Modalities & Architectures
Each sensory branch extracts highly specialized features before reaching the fusion brain. The exact neural architectures are defined as follows:

- **Emotion**: `convnext_small` (via `timm`) augmented with a custom **CBAM** (Convolutional Block Attention Module) for deep facial affect and micro-expression detection.
- **Environment**: `resnet50` (pretrained vision backbone) for processing the contextual layout of the scene (e.g., distinguishing a hospital bed from an office environment).
- **Health**: `Custom 1D CNN` (`ConvBlock1D` arrays with kernel=7) designed to process multi-channel physiological or physical indicator signals over time.
- **Gesture**: `efficientnet_b0` (via `torchvision.models`) functioning as a Confidence-Gated Gesture Encoder to reliably track hand signals like "Help" or "Stop" while masking false positives.
- **Speech**: `facebook/wav2vec2-base` (via HuggingFace Transformers) processing audio on a 1.0s sliding window to capture real-time spoken distress commands.

---

## 🏗 Industrial Architecture & Data Flow

```mermaid
graph TD
    %% Input Layer
    Sensors[IP Webcam / Local Microphone] -->|Raw Video/Audio Streams| Pre[Preprocessing & 1.0s Sliding Window]
    
    %% Branch Layers
    subgraph "Modality Feature Extraction"
        Pre -->|Faces| EMO[Emotion Branch]
        Pre -->|Context| ENV[Environment Branch]
        Pre -->|Signals| HEA[Health Branch]
        Pre -->|Movement| GES[Gesture Branch]
        Pre -->|Vocal| SPE[Speech Branch]
    end
    
    %% Fusion Layer
    EMO & ENV & GES & SPE & HEA -->|512d Embeddings + Confidence Gates| FUS[Attention-Masked Fusion Brain]
    
    %% Stabilization Layer
    FUS -->|Raw Probability Logits| RL[RL Smoother: Proximal Policy Optimization]
    RL -->|Temporally Stable Probabilities| DEC[Final 8-Class Action Prediction]
    
    %% Iterative Learning Pipeline
    DEC -->|Detects Low Confidence| BM[Buffer Manager]
    BM -->|Saves Edge-Case Frames and Audio| DB[(Persistent Data Buffer)]
    DB -->|Offline Fine-Tuning Scripts| TRAIN[Training Workflow]
    TRAIN -->|Saves v2, v3 Weights| FS[System File Storage]
    FS -->|Auto-Versioning Runtime Load| EMO & FUS
```

---

## ⚙️ Complete Setup & Execution Flow

Follow this exact workflow to deploy the framework on your local machine or server.

### Step 1: Clone the Repository
```bash
git clone https://github.com/2023priyanshubhargav-cpu/dl-project.git
cd dl-project
```

### Step 2: Environment Setup
It is highly recommended to isolate the project within a virtual environment.
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install core Neural Network engines (CUDA 11.8 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required Vision, Audio, and Utility libraries
pip install opencv-python numpy pillow sounddevice librosa transformers timm requests
```
*(Note: Upon first execution, the `transformers` library will require an internet connection to download the Wav2Vec2 weights).*

### Step 3: Hardware Sensor Configuration (IP Webcam)
The system natively supports the **IP Webcam** app (Android/iOS) to utilize your mobile phone as an untethered biometric sensor.
1. Download and launch **IP Webcam** on your mobile device.
2. Press "Start server" at the bottom of the app.
3. Note the IP address displayed on your phone screen (e.g., `192.168.1.5:8080`).
4. Open `realtime_fusion_8cls.py` and configure the endpoint on line ~88:
    ```python
    USE_IP_WEBCAM_AUDIO = True
    IP_WEBCAM_IP        = "192.168.1.5" # Update this to match your phone
    IP_WEBCAM_PORT      = 8080
    ```

### Step 4: Run the Real-Time Inference Engine
Initialize the 5-branch multimodal fusion interface:
```bash
python3 realtime_fusion_8cls.py
```
**Runtime Controls:**
- Press **'Q'** to safely shut down the system. This gracefully terminates audio threads and writes any pending buffer data to disk.

---

## 🔄 The Full Iterative Learning Process

The true strength of MPMIS is its ability to autonomously collect training data when it fails, allowing developers to create rapidly evolving model versions.

### Phase 1: Autonomous Edge-Case Collection
While `realtime_fusion_8cls.py` is running, the **Buffer Manager** analyzes the confidence score of the Fusion Brain. If the prediction is uncertain (e.g., < 75% confidence on an "Emergency" action), the system captures the raw video frame and the 1.0s audio chunk and saves it locally in the `/buffers/[modality]/` directories.

### Phase 2: Supervised Label Verification
Developers can periodically check the `/buffers/` directory to review the edge cases that confused the system. Mislabelled data can be manually moved or corrected.

### Phase 3: Fine-Tuning Execution
Run the dedicated training scripts to fine-tune specific modalities or the overarching fusion brain on the newly collected edge cases:
```bash
python3 train_emotion_finetune.py
# or
python3 train_fusion_v3.py
```

### Phase 4: Auto-Versioning Deployment
The fine-tuning scripts are programmed to save weights sequentially (e.g., `emotion_model_full_v2.pth`, `best_fusion_model_8cls_v3.pt`).
The primary `realtime_fusion_8cls.py` script utilizes a `get_latest_model()` dynamic loader. Upon your next execution, the system will **automatically detect and load** the highest-versioned weights, closing the iterative loop with zero manual code changes.

---

## 📂 Comprehensive Module Catalog

| Module | Filename | Function |
| :--- | :--- | :--- |
| **Inference Engine** | `realtime_fusion_8cls.py` | The core runtime loop. Initializes all branches, executes the audio/video streams, applies the gating thresholds, and orchestrates the Fusion and RL models. |
| **Data Orchestration**| `buffer_manager.py` | Governs the memory management. Analyzes runtime uncertainty, manages IO operations, and organizes edge-case files for the Iterative Loop. |
| **Decision Smoothing**| `ppo_inference.py` | Reinforcement Learning agent. Smooths the raw probability distributions generated by the Fusion Brain to prevent rapid output flickering. |
| **Audio Streamer** | `ip_audio_streamer.py` | Custom thread-safe HTTP streaming client to reliably fetch and decode mobile audio streams without blocking the main vision loop. |
| **Training Pipeline** | `train_fusion_v3.py`, `train_*_finetune.py` | Automated PyTorch training loops designed to ingest the `/buffers/` data and export `vX+1` model weights. |
| **JIT Exporter** | `export_embedding_models.py` | Converts dynamic `.pth` models into highly optimized, production-ready TorchScript (`.pt`) files for low-latency inference. |
| **Jupyter Research** | `Emotion.ipynb`, `Speech.ipynb`, `Gesture.ipynb` | Origin training notebooks containing the foundational research, exploratory data analysis, and initial backbone architecture definitions. |
