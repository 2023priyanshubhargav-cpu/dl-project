# 🔄 CONTINUAL LEARNING SYSTEM — COMPLETE IMPLEMENTATION PLAN
**Date: April 25, 2026**  
**Project: Multimodal Patient Monitoring with Automatic Fine-Tuning**

---

## 📋 TABLE OF CONTENTS
1. [System Overview](#system-overview)
2. [Core Concepts](#core-concepts)
3. [Data Collection Strategy](#data-collection-strategy)
4. [Model Classes & Thresholds](#model-classes--thresholds)
5. [Collection Pipeline](#collection-pipeline)
6. [Retraining Pipeline](#retraining-pipeline)
7. [Model Versioning](#model-versioning)
8. [Integration Points](#integration-points)
9. [File Structure](#file-structure)
10. [Workflow Timeline](#workflow-timeline)

---

## 🎯 SYSTEM OVERVIEW

### Purpose
Deploy a patient monitoring system that **learns from real patients** after initial deployment. The system automatically collects high-confidence examples from live inference, periodically retrains individual models on this patient-specific data, retrains the fusion layer, and updates RL agents—all while maintaining the ability to rollback to previous versions if quality degrades.

### Key Principle: **Prevent Catastrophic Forgetting**
When retraining on new data, the model must not forget what it learned on generic data. Solution: **Replay-Based Continual Learning**.
- Mix ~10-15% new patient data + ~85-90% original generic data during each retraining cycle
  - Example: 50 new images + 300-400 original images per batch
  - Ensures patient-specific learning WITHOUT overfitting or forgetting generic knowledge
- Fine-tune for 10 epochs (warm-start, not from scratch)
- Save versioned models (v1, v2, v3...)

---

## 🧠 CORE CONCEPTS

### Confidence Threshold Strategy
**Threshold: 90%** (for all modalities, all classes)
- Only save data if model prediction probability > 90%
- Prevents noisy/ambiguous frames from entering the training buffer
- Reduces manual verification overhead

### Per-Class Minimum Thresholds
Each class requires a **minimum number of collected images** before retraining triggers:

| Model | Class | Min Images | Modality Type |
|-------|-------|-----------|---------------|
| **Emotion** | Angry | 50 | Image |
| **Emotion** | Happy | 50 | Image |
| **Emotion** | Neutral | 50 | Image |
| **Emotion** | Surprise | 50 | Image |
| **Emotion** | Disgust | 50 | Image (augmented after 5 complete) |
| **Emotion** | Fear | 50 | Image (MANUAL capture ONLY - after 5 emotions done) |
| **Emotion** | Sad | 50 | Image (MANUAL capture ONLY - after 5 emotions done) |
| **Gesture** | All 10 classes | 40 each | Image |
| **Speech** | All 10 commands | 30 each | Audio |
| **Health** | Baseline/Stress/Amusement/Meditation | N/A for live collection | No raw health collection for now; health embeddings are provided directly to fusion when available |
| **Environment** | Dynamic top 5 live classes | 25 each | Image (top 5 selected by highest live-buffer counts, with augmentation) |

### Augmentation Strategy
- **Emotion Disgust**: Flip, rotate (±15°), brightness (±0.2), contrast (±0.2)
- **Environment top-5 buffer completion**: If a selected top-5 class has <25 live images, augment from that class's own buffer images until 25.
- **Environment generic replay**: Remaining replay/generic examples come from the original/main dataset, with augmentation if needed, and are marked as generic/non-patient data.
- **Health**: NO raw live collection for now. Health is handled as embeddings for fusion because no real health sensor stream is available.
- **Speech**: NO augmentation (during collection only)

### Metadata Strategy
Every saved live sample should have a `.meta.json` sidecar with confidence, modality, class, source, timestamp, whether it is augmented, and whether it is patient-specific. This is not required to train from folder labels, but it is important for audit, rollback, filtering bad samples, and debugging model drift.

---

## 📊 DATA COLLECTION STRATEGY

### Stage 1: Automatic Collection (Live Inference)
**Trigger**: `realtime_fusion_8cls.py` is running with RL smoother active

**Process**:
1. Live inference runs continuously
2. For each frame processed:
   - Extract embeddings from Emotion, Gesture, Speech, Environment models
   - Get logits and compute softmax probabilities
   - For each modality, check: `max(softmax) > 90%`?
   - If YES → Save frame + metadata to corresponding class buffer
   - If NO → Discard frame (too uncertain)

**Directory Structure**:
```
buffers/
├── emotion/
│   ├── angry/
│   │   ├── emotion_patient_X_frame_0001.jpg
│   │   ├── emotion_patient_X_frame_0002.jpg
│   │   └── ... (target: 50 images)
│   ├── happy/
│   ├── neutral/
│   ├── surprise/
│   ├── disgust/         (auto-saved when confident)
│   ├── fear/            (manual capture phase)
│   └── sad/             (manual capture phase)
├── gesture/
│   ├── help/
│   ├── stop/
│   └── ... (all 10 classes)
├── speech/
│   ├── yes/
│   ├── no/
│   └── ... (all 10 commands, as .wav files)
├── health/
│   └── no live raw health collection in current setup
└── environment/
    ├── dynamically-created-class-1/
    ├── dynamically-created-class-2/
    └── ... (top 5 most frequent classes selected per cycle)
```

### Stage 2: Monitor & Alert (Conditional Manual Collection for Rare Emotions)
**Trigger ONLY when**: ALL of [Angry, Happy, Neutral, Surprise, Disgust] = 50 images each

**IMPORTANT**: Angry, Happy, Neutral, Surprise, Disgust collect automatically during LIVE INFERENCE. Fear and Sad MUST be collected manually ONLY AFTER the first 5 emotions reach 50 images each.

**Code Behavior**:
```python
# Check every 100 frames
if (emotion_buffers['angry'] >= 50 AND
    emotion_buffers['happy'] >= 50 AND
    emotion_buffers['neutral'] >= 50 AND
    emotion_buffers['surprise'] >= 50 AND
    emotion_buffers['disgust'] >= 50 AND
    emotion_buffers['fear'] < 50 AND
    emotion_buffers['sad'] < 50):
    
    PAUSE_LIVE_COLLECTION()
    PRINT_ALERT():
        print("\n" + "="*60)
        print("✓ EMOTION COLLECTION PHASE 1 COMPLETE (5/7 emotions ready)")
        print("="*60)
        print("\nStatus:")
        print("  ✓ Angry:    50/50 images")
        print("  ✓ Happy:    50/50 images")
        print("  ✓ Neutral:  50/50 images")
        print("  ✓ Surprise: 50/50 images")
        print("  ✓ Disgust:  50/50 images (augmented)")
        print("  ✗ Fear:     0/50 images  (MANUAL CAPTURE NEEDED)")
        print("  ✗ Sad:      0/50 images  (MANUAL CAPTURE NEEDED)")
        print("\n📸 ENTER MANUAL CAPTURE MODE")
        print("\n[STEP 1] Show FEAR expressions to camera")
        print("  - Make fearful face (eyes wide, eyebrows raised)")
        print("  - System auto-saves frames where confidence > 90%")
        print("  - Collect until: buffers/emotion/fear/ has 50 images")
        print("  - Press SPACEBAR when complete")
        print("\n[STEP 2] Show SAD expressions to camera")
        print("  - Make sad face (frown, downward gaze)")
        print("  - System auto-saves frames where confidence > 90%")
        print("  - Collect until: buffers/emotion/sad/ has 50 images")
        print("  - Press SPACEBAR when complete")
        print("\n" + "="*60)
        print("Press ENTER to start FEAR manual capture...")
        print("="*60 + "\n")
```

### Stage 3: Manual Capture (User-Driven — ONLY for Fear & Sad)
**CRITICAL**: This stage ONLY triggers after first 5 emotions (Angry, Happy, Neutral, Surprise, Disgust) reach 50 images each.

**Fear Capture**:
1. Live camera shows: "FEAR CAPTURE ACTIVE — Confidence > 90% required"
2. User makes fearful expressions to camera
3. System checks each frame: `emotion_model.predict(frame) == 'fear' AND confidence > 90%?`
4. If YES → Auto-save to `buffers/emotion/fear/[timestamp].jpg`
5. Monitor: Live counter shows "Fear: X/50 images"
6. When counter reaches 50 → User presses SPACEBAR → Move to SAD capture

**Sad Capture**:
1. Live camera shows: "SAD CAPTURE ACTIVE — Confidence > 90% required"
2. User makes sad expressions to camera
3. System checks each frame: `emotion_model.predict(frame) == 'sad' AND confidence > 90%?`
4. If YES → Auto-save to `buffers/emotion/sad/[timestamp].jpg`
5. Monitor: Live counter shows "Sad: X/50 images"
6. When counter reaches 50 → User presses SPACEBAR → Manual capture COMPLETE

**Expected Collection Time**: ~2-3 minutes per emotion (user makes natural expressions; system captures only high-confidence peaks)

**After Manual Capture Complete**: All 7 emotion classes ready → Retraining pipeline auto-triggers

---

## 🏷️ MODEL CLASSES & THRESHOLDS

### Emotion Model (7 Classes)
```
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_EMOTION_CLASSES = 7
PER_CLASS_MIN = 50  # images each
TOTAL_EMOTION_BUFFER = 350 images (7 × 50)
CONFIDENCE_THRESHOLD = 90%
```

### Gesture Model (10 Classes)
```
GESTURE_CLASSES = ['help', 'stop', 'yes', 'no', 'calm', 'attention', 
                   'emergency', 'suspicious', 'cancel', 'unknown']
NUM_GESTURE_CLASSES = 10
PER_CLASS_MIN = 40  # images each
TOTAL_GESTURE_BUFFER = 400 images (10 × 40)
CONFIDENCE_THRESHOLD = 90%
```

### Speech Model (10 Commands)
```
SPEECH_COMMANDS = ['yes', 'no', 'stop', 'go', 'left', 'right', 
                   'up', 'down', 'on', 'off']
NUM_SPEECH_CLASSES = 10
PER_CLASS_MIN = 30  # audio samples each
TOTAL_SPEECH_BUFFER = 300 samples (10 × 30)
CONFIDENCE_THRESHOLD = 90%
MODEL_INPUT_SAMPLE_RATE = 16000  # resample live/IP Webcam audio before Wav2Vec inference
```

### Health Model (4 Classes, Fusion Embeddings Only For Now)
```
HEALTH_CLASSES = ['baseline', 'stress', 'amusement', 'meditation']
NUM_HEALTH_CLASSES = 4
RAW_LIVE_COLLECTION = False  # no real health sensor stream available
FUSION_INPUT = "health embeddings supplied directly when available"
```

### Environment Model (Dynamic Top 5 Live Classes)
```
ENV_TOP_CLASSES = top_5_by_count(buffers/environment/*)
NUM_ENV_CLASSES_PER_CYCLE = 5
PER_CLASS_MIN = 25  # live + buffer augmentation for selected top 5
CONFIDENCE_THRESHOLD = 90%
BUFFER_AUGMENTATION_SOURCE = "same class's live buffer images"
GENERIC_REPLAY_SOURCE = "main/original environment dataset"
```

---

## 🔄 COLLECTION PIPELINE

### Collection Flow Diagram
```
START LIVE INFERENCE
    ↓
[realtime_fusion_8cls.py running]
    ↓
For each frame:
    ├─ Extract embeddings (Emotion, Gesture, Speech, Env)
    ├─ Get predictions + confidence scores
    ├─ For each modality:
    │   └─ if confidence > 90%:
    │       └─ Save frame to buffers/[modality]/[class]/
    └─ Continue
    ↓
MONITOR BUFFER STATUS (every 100 frames)
    ├─ if Emotion (5/7 classes) meet min threshold:
    │   └─ PAUSE & ASK FOR MANUAL CAPTURE
    │       ├─ Capture FEAR until threshold met
    │       ├─ Capture SAD until threshold met
    │       └─ Resume monitoring
    │
    ├─ if Gesture (all 10) meet min threshold: ✓
    ├─ if Speech (all 10) meet min threshold: ✓
    ├─ Health raw collection skipped; fusion uses supplied health embeddings when available
    └─ if Environment (dynamic top 5) meet min threshold after buffer augmentation: ✓
    ↓
WHEN ALL INDIVIDUAL MODELS READY
    └─ TRIGGER RETRAINING PIPELINE
```

---

## 🔄 RETRAINING PIPELINE

### Phase 1: Individual Model Retraining (Parallel)
**When**: All per-class minimums met for each modality

**Sequence** (can run in parallel):
1. `train_emotion_finetune.py`
   - Load: `emotion_model_full.pth` (original v1)
   - Data: Mix ~15% from `buffers/emotion/` + ~85% from original training set
     - Specifically: 50 new images (all 7 emotions) + ~300-350 original samples
   - Epochs: 10 (warm-start fine-tune, low learning rate)
   - Save: `emotion_model_full_v2.pth` (state dict for future retraining)
   - Save: `emotion_model_full_v2.pt` (TorchScript for inference)

2. `train_gesture_finetune.py`
   - Load: `gesture_model.pth`
   - Data: Mix ~15% from `buffers/gesture/` (400 images) + ~85% original
     - Specifically: 400 new images + ~2300 original samples
   - Save: `gesture_model_v2.pth` + `gesture_model_v2.pt`

3. `train_speech_finetune.py`
   - Load: `speech_model_new.pth`
   - Data: Real `.wav` samples from `buffers/speech/`
   - Preprocessing: resample all live/IP Webcam audio to 16 kHz / 16000 samples before Wav2Vec
   - Save: `speech_model_v2.pth`
   - Note: No dummy/random audio is allowed. If original speech replay data is unavailable, the script must say so clearly instead of inventing data.

4. `train_env_finetune.py`
   - Load: `environment_model_full.pth`
   - Data: dynamic top 5 classes from `buffers/environment/`
   - If any selected top-5 class has <25 buffer images, augment from that class's own buffer images until 25.
   - Generic replay/remaining examples come from the main/original environment dataset and are tagged as generic/non-patient.
   - Save: `environment_model_full_v2.pth` + `environment_model_full_v2.pt`

**No Dummy Training Rule**: Retraining scripts must never train on random tensors or fake labels. If real data, original replay data, embeddings, or model checkpoints are missing, the script should fail loudly and write the reason to the continual-learning log.

### Phase 2: Fusion Model Retraining
**When**: All individual models done with Phase 1

**Process**:
- `train_fusion_finetune.py`
  - Load: `best_fusion_model_8cls.pth`
  - Load: NEW individual models (`emotion_v2.pt`, `gesture_v2.pt`, etc.)
  - Data: Collect NEW fusion pairs from live inference with v2 models
  - Save: `best_fusion_model_8cls_v2.pth` + `best_fusion_model_8cls_v2.pt`

### Phase 3: RL Agent Retraining
**When**: Fusion model done with Phase 2

**Process**:
- `train_rl_agents_finetune.py`
  - Load: `ppo_smoother.zip` (v1)
  - Load: NEW fusion model (`best_fusion_model_8cls_v2.pt`)
  - Generate trajectories from live inference with v2 fusion
  - PPO update with new data
  - Save: `ppo_smoother_v2.zip`

---

## 📦 MODEL VERSIONING

### File Naming Convention
```
# ORIGINAL (v0, not versioned)
emotion_model_full.pth              # State dict
emotion_model_scripted.pt           # TorchScript

# AFTER FIRST RETRAIN (v1)
emotion_model_full_v2.pth           # State dict for next retrain
emotion_model_full_v2.pt            # TorchScript for inference
emotion_model_scripted_v2.pt        # Alternative naming

# AFTER SECOND RETRAIN (v2)
emotion_model_full_v3.pth
emotion_model_full_v3.pt

# Pattern
[model_name]_v[N].pth              # State dict
[model_name]_v[N].pt               # TorchScript
```

### Rollback Strategy
Keep **at least 2 versions** at all times:
- Current version (v_N)
- Previous version (v_{N-1}) for immediate rollback
- Delete v_{N-2} and older to save disk space

**Rollback Trigger**: If accuracy drops > 5% in validation set, revert to v_{N-1}:
```bash
# Rollback in realtime_fusion_8cls.py path block
EMOTION_MODEL_PATH = os.path.join(BASE, "emotion_model_full_v2.pt")  # revert from v3 to v2
```

---

## 🔗 INTEGRATION POINTS

### 1. Real-Time Inference Script
**File**: `realtime_fusion_8cls.py`

**Modifications**:
- Add continual learning buffer manager
- Monitor per-class counts
- Trigger manual capture prompts
- Save high-confidence frames
- Switch model paths for testing

**Path Block** (editable for testing):
```python
BASE = os.path.dirname(os.path.abspath(__file__))

# Current production versions
EMOTION_MODEL_PATH  = os.path.join(BASE, "emotion_model_full_v2.pt")
GESTURE_MODEL_PATH  = os.path.join(BASE, "gesture_model_v2.pt")
SPEECH_MODEL_PATH   = os.path.join(BASE, "speech_model_v2.pt")
ENV_MODEL_PATH      = os.path.join(BASE, "environment_model_full_v2.pt")
FUSION_MODEL_PATH   = os.path.join(BASE, "best_fusion_model_8cls_v2.pt")

# (To switch: just change v2 → v3, etc.)
```

### 2. Buffer Manager
**New File**: `buffer_manager.py`

**Responsibilities**:
- Track buffer file counts per class
- Compute when per-class thresholds are met
- Trigger retraining pipeline
- Handle augmentation for specific classes

**Key Methods**:
```python
class BufferManager:
    def get_status(self):
        """Return current counts per class per modality"""
        
    def is_emotion_ready_for_retrain(self):
        """Check if all 7 emotion classes >= 50 images"""
        
    def prompt_for_manual_capture(self):
        """Print instructions for Fear/Sad capture"""
        
    def save_frame(self, frame, modality, class_name, confidence):
        """Save frame to buffer if conditions met"""
        
    def augment_and_save_environment(self):
        """Augment remaining env images to ~140 total"""
        
    def augment_and_save_disgust(self):
        """Augment disgust samples if not enough collected naturally"""
```

### 3. Retraining Scripts
**Files**:
- `train_emotion_finetune.py`
- `train_gesture_finetune.py`
- `train_speech_finetune.py`
- `train_env_finetune.py`
- `train_fusion_finetune.py`
- `train_rl_agents_finetune.py`

**Common Template**:
```python
def retrain(num_epochs=5):
    # 1. Load original model (state dict)
    model = load_original_model()
    
    # 2. Load buffer data (15%)
    buffer_data = load_buffer_data()
    
    # 3. Load original training data (85%)
    original_data = load_original_data()
    
    # 4. Mix datasets (15% new, 85% original)
    combined_loader = mix_dataloaders(buffer_data, original_data, ratio=0.15)
    
    # 5. Fine-tune with lower learning rate
    optimizer = Adam(lr=1e-4)  # 10x lower than initial training
    for epoch in range(num_epochs):
        for batch in combined_loader:
            loss = model(batch)
            optimizer.step()
    
    # 6. Save BOTH versions
    torch.save(model.state_dict(), "model_full_v2.pth")      # For next retrain
    model_scripted = torch.jit.script(model)
    model_scripted.save("model_full_v2.pt")                  # For inference
    
    # 7. Validate on hold-out test set
    test_acc = validate(model, test_set)
    print(f"Test Accuracy (v2): {test_acc:.2%}")
    
    return test_acc
```

---

## 🗂️ FILE STRUCTURE

```
emotion_project/
├── realtime_fusion_8cls.py          (Modified: add buffer management)
├── buffer_manager.py                (NEW)
├── ppo_inference.py                 (Existing: RL smoother)
│
├── train_emotion_finetune.py         (NEW)
├── train_gesture_finetune.py         (NEW)
├── train_speech_finetune.py          (NEW)
├── train_env_finetune.py             (NEW)
├── train_fusion_finetune.py          (NEW)
├── train_rl_agents_finetune.py       (NEW)
│
├── models/
│   ├── emotion_model_full.pth        (Original v1)
│   ├── emotion_model_full_v2.pth     (After retrain v2)
│   ├── emotion_model_full_v2.pt      (TorchScript inference v2)
│   ├── gesture_model.pth             (Original)
│   ├── gesture_model_v2.pth          (After retrain)
│   ├── ... (all other models)
│   ├── best_fusion_model_8cls.pth
│   ├── best_fusion_model_8cls_v2.pth
│   └── ppo_smoother_v2.zip
│
├── buffers/
│   ├── emotion/
│   │   ├── angry/         (target: 50 images)
│   │   ├── disgust/       (target: 50 images, augmented if needed)
│   │   ├── fear/          (target: 50 images, manual capture)
│   │   ├── happy/         (target: 50 images)
│   │   ├── neutral/       (target: 50 images)
│   │   ├── sad/           (target: 50 images, manual capture)
│   │   └── surprise/      (target: 50 images)
│   ├── gesture/           (10 classes × 40 images)
│   ├── speech/            (10 commands × 30 samples)
│   ├── health/            (no raw live collection in current setup)
│   └── environment/       (dynamic top 5 × 25, plus generic replay from main dataset)
│
├── datasets/              (Original training data)
│   ├── fer2013/
│   ├── raf-db-dataset/
│   ├── ferplus/
│   ├── places365/
│   ├── places365_filtered/
│   └── wesad/
│
└── CONTINUAL_LEARNING_PLAN.md        (THIS FILE)
```

---

## ⏱️ WORKFLOW TIMELINE

### Day 1: Start Live Monitoring
```
T=0:00
├─ Start: python3 realtime_fusion_8cls.py
├─ System runs live inference (collects Emotion, Gesture, Speech, Env)
├─ Frames > 90% confidence auto-saved to buffers
└─ Continue for 2-4 hours or until some buffers fill

T=2:00 - 4:00
├─ Emotion: Angry ✓, Happy ✓, Neutral ✓, Surprise ✓, Disgust ✓
├─ Emotion: Fear ✗ (only 15/50), Sad ✗ (only 12/50)
├─ [SYSTEM PAUSES & ALERTS FOR MANUAL CAPTURE]
└─ Continue other modalities...
```

### Day 2: Manual Capture Phase
```
T=4:00 - 4:30
├─ User shows FEAR expressions to camera
├─ System auto-saves frames where emotion_confidence > 90% && pred == 'fear'
├─ Collect ~50 images
├─ Press SPACEBAR
└─ Continue...

T=4:30 - 5:00
├─ User shows SAD expressions
├─ Collect ~50 images
├─ System now has all 7 emotion classes ready ✓
└─ All other modalities also ready ✓

T=5:00 → RETRAIN TRIGGER
```

### Day 2: Retraining Phase
```
T=5:00 - 5:20 [PARALLEL: Individual Models]
├─ train_emotion_finetune.py (10 epochs)
├─ train_gesture_finetune.py (10 epochs)
├─ train_speech_finetune.py (10 epochs)
├─ train_env_finetune.py (10 epochs)
├─ health raw retraining skipped unless real health data becomes available
└─ All save: [model]_v2.pth + [model]_v2.pt

T=5:20 - 5:30 [SEQUENTIAL: Fusion]
├─ train_fusion_finetune.py
├─ Load NEW v2 models
├─ Fine-tune fusion layer
└─ Save: best_fusion_model_8cls_v2.pth + .pt

T=5:30 - 5:40 [RL AGENT]
├─ train_rl_agents_finetune.py
├─ Retrain PPO on NEW fusion v2
└─ Save: ppo_smoother_v2.zip

T=5:40 - 6:00 [VALIDATION]
├─ Test v2 models on hold-out test set
├─ Compare accuracy: v1 vs v2
├─ If v2 > v1: Deploy (update realtime_fusion_8cls.py paths)
├─ If v2 < v1: Rollback (keep using v1 paths)
└─ Log results
```

### Day 3+: Continue Learning Cycle
```
T=6:00 onwards
├─ Switch realtime_fusion_8cls.py to use v2 models
├─ Continue live inference with improved models
├─ New buffers start filling for next cycle (v3)
├─ Repeat collection → retrain → validate → deploy cycle
└─ Every 1-2 weeks for continuous improvement
```

---

## ⚙️ CONFIGURATION & HYPERPARAMETERS

### Confidence Threshold
```python
CONFIDENCE_THRESHOLD_GLOBAL = 0.90  # 90%
```

### Per-Class Minimums
```python
EMOTION_MIN_PER_CLASS = 50
GESTURE_MIN_PER_CLASS = 40
SPEECH_MIN_PER_CLASS = 30
HEALTH_MIN_PER_CLASS = 30
ENV_TARGET_CLASSES = 5
ENV_MIN_PER_CLASS = 25
```

### Retraining Hyperparameters
```python
RETRAIN_EPOCHS = 10
LEARNING_RATE = 1e-4  # 10x lower than initial training
BATCH_SIZE = 32

# Data Mix (NEW RATIO - NOT 50/50)
NEW_DATA_RATIO = 0.15  # ~15% new patient data
ORIGINAL_DATA_RATIO = 0.85  # ~85% original generic data

# Example calculation:
# If buffer has 50 new emotion images:
#   50 / (50 + x) = 0.15
#   x = 280 original samples needed
# Every epoch sees: 50 new + 280 original = 330 samples
```

### Model Save Locations
```python
MODELS_DIR = "./models/"
BUFFERS_DIR = "./buffers/"
DATASETS_DIR = "./datasets/"
```

---

## ✅ SUCCESS CRITERIA

### Collection Phase
- [ ] Emotion: All 7 classes >= 50 images (Angry, Happy, Neutral, Surprise, Disgust, Fear, Sad)
- [ ] Gesture: All 10 classes >= 40 images each
- [ ] Speech: All 10 commands >= 30 samples each
- [ ] Health: raw live collection intentionally skipped; fusion uses health embeddings when available
- [ ] Environment: dynamic top 5 most frequent live classes, each >=25 after buffer augmentation

### Retraining Phase
- [ ] All individual models retrain successfully (loss decreases, no NaNs)
- [ ] Fusion model retrains on mixed data from v2 individual models
- [ ] RL agent retrains and stabilizes on new fusion output
- [ ] All v2 models save as BOTH `.pth` (state dict) and `.pt` (TorchScript)

### Validation Phase
- [ ] v2 models evaluated on hold-out test set
- [ ] Accuracy maintained or improved (no drop > 5%)
- [ ] If quality good → Deploy v2 models in production
- [ ] If quality bad → Rollback to v1 models, analyze why

### Production Phase
- [ ] realtime_fusion_8cls.py uses v2 models
- [ ] New buffers start filling for next cycle
- [ ] System continues autonomous learning loop
- [ ] Version history preserved for auditability

---

## 🚨 ERROR HANDLING & ROLLBACK

### If Retrain Fails
```
Error: train_emotion_finetune.py crashes
→ Check buffer data integrity
→ Verify original training data available
→ Retry with smaller batch size
→ If still fails: Skip this emotion retrain, continue with others
```

### If v2 Accuracy Drops
```
Validation: v2_accuracy < v1_accuracy - 5%
→ Flag model as "unsafe for deployment"
→ Keep using v1 models in production
→ Analyze: What went wrong? Overfitting? Bad augmentation?
→ Collect more diverse buffer data
→ Retry retraining with lower learning rate
```

### If Storage Runs Out
```
Disk Space: <10% free
→ Archive oldest version (v_{N-3} or older)
→ Keep only: current version + previous version
→ Move archived versions to external storage if needed
```

---

## 📝 LOGGING & MONITORING

### Per-Collection Cycle
Log file: `logs/collection_cycle_001.txt`
```
[2026-04-25 10:00:00] Collection started
[2026-04-25 10:05:00] Emotion/Angry: 10/50 images
[2026-04-25 10:10:00] Emotion/Angry: 20/50 images
...
[2026-04-25 14:00:00] ✓ All emotion classes ready
[2026-04-25 14:00:00] ✓ All gesture classes ready
...
[2026-04-25 14:05:00] COLLECTION COMPLETE — Starting retraining
```

### Per-Retrain Cycle
Log file: `logs/retrain_cycle_001.txt`
```
[2026-04-25 14:05:00] Retraining cycle started
[2026-04-25 14:07:00] emotion_finetune: Epoch 1/5 — Loss: 0.45, Val Acc: 92.3%
[2026-04-25 14:08:00] emotion_finetune: Epoch 2/5 — Loss: 0.38, Val Acc: 93.1%
...
[2026-04-25 14:15:00] emotion_finetune: COMPLETE — v2 saved (93.5% accuracy)
[2026-04-25 14:16:00] gesture_finetune: COMPLETE — v2 saved (91.2% accuracy)
...
[2026-04-25 14:30:00] fusion_finetune: COMPLETE — v2 saved (88.9% accuracy)
[2026-04-25 14:35:00] rl_agents_finetune: COMPLETE — v2 saved
[2026-04-25 14:40:00] Validation: v1=87.2%, v2=88.1% — ✓ DEPLOY v2
```

---

## 📚 ACADEMIC FRAMING (For Your Rubric)

### Continual Learning Variant Used
**Class-Incremental Learning (CIL) + Replay-Based Approach**

### Why This Approach?
1. **Prevents Catastrophic Forgetting**: Mixing new patient data with generic data ensures old knowledge is retained
2. **Scalable**: Works without storing infinite data—only keep recent buffers
3. **Patient-Specific Adaptation**: Model personalizes to individual patient characteristics over time
4. **Healthcare-Safe**: Confidence thresholding prevents model learning from its own mistakes

### Novel Aspects
1. **Per-Class Thresholds**: Not all classes collect equally fast; manual capture for rare emotions
2. **Multimodal Synchronization**: All 5 modalities must reach thresholds before retraining (prevents modality drift)
3. **RL Agent Retraining**: PPO agents retrain on patient-adapted fusion outputs for better smoothing

---

## 🔧 NEXT STEPS (When User Says "YES")

1. Create `buffer_manager.py` for automated buffer tracking
2. Modify `realtime_fusion_8cls.py` to integrate buffer management
3. Create 6 retraining scripts (emotion, gesture, speech, env, fusion, rl)
4. Test collection pipeline with real or recorded samples only; no dummy/random training data
5. Run first full cycle (collect → retrain → validate → deploy)
6. Document results and accuracy improvements

---

**Document Version**: 1.0  
**Last Updated**: April 25, 2026  
**Status**: READY FOR IMPLEMENTATION
