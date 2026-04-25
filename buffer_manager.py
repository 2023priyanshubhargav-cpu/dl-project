import os
import cv2
import numpy as np
import json
import time
import uuid
import torch
import wave
import subprocess
import sys

class BufferManager:
    def __init__(self, base_dir="./buffers", environment_classes=None):
        self.base_dir = base_dir
        self.environment_classes = set(environment_classes) if environment_classes else None
        self.thresholds = {
            'emotion': {
                'angry': 50, 'disgust': 50, 'happy': 50, 
                'neutral': 50, 'surprise': 50, 'fear': 50, 'sad': 50
            },
            'gesture': {c: 40 for c in ['help', 'stop', 'yes', 'no', 'calm', 
                                        'attention', 'emergency', 'suspicious', 'cancel', 'unknown']},
            'speech': {c: 30 for c in ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']},
            'health': {c: 30 for c in ['baseline', 'stress', 'amusement', 'meditation']},
            'environment': {}  # DYNAMIC - Classes created on-the-fly as they appear
        }
        self.confidence_threshold = 90.0
        self.speech_conf_threshold = 80.0 # Lowered to collect more data
        self.env_class_threshold = 25  # Each environment class needs 25 images minimum
        self.env_target_classes = 5  # Dynamically select the 5 most frequent live classes
        
        # State tracking
        self.phase_1_complete = False
        self.manual_capture_mode = None 
        
        # Iterative Versioning
        self.version_file = os.path.join(self.base_dir, "model_versions.json")
        self._load_versions()

        # Auto-Retraining State tracking
        self.retrain_status = {
            'emotion': 'pending', 
            'gesture': 'pending', 
            'speech': 'pending', 
            'environment': 'pending',
            'fusion': 'pending'
        }   # 'pending', 'running', 'completed'
        self.retrain_processes = {}
        self.retrain_log_handles = {}
        self.retrain_scripts = {
            'emotion': 'train_emotion_finetune.py',
            'gesture': 'train_gesture_finetune.py',
            'speech': 'train_speech_finetune.py',
            'environment': 'train_env_finetune.py',
            'fusion': 'train_fusion_finetune.py'
        }
        
        self._init_directories()

    def _init_directories(self):
        """Create the directory structure for all modalities and classes."""
        for modality, classes in self.thresholds.items():
            for class_name in classes.keys():
                os.makedirs(os.path.join(self.base_dir, modality, class_name), exist_ok=True)

    def get_count(self, modality, class_name=None):
        """Count files for a specific class or total for modality if class_name is None."""
        if class_name:
            # Count specifically for one class folder
            path = os.path.join(self.base_dir, modality, class_name)
            if not os.path.exists(path): return 0
            return len([f for f in os.listdir(path) if not f.endswith(".json")])
        else:
            # Total count for the whole modality (summing all class folders)
            mod_path = os.path.join(self.base_dir, modality)
            if not os.path.exists(mod_path): return 0
            total = 0
            for root, dirs, files in os.walk(mod_path):
                total += len([f for f in files if not f.endswith(".json")])
            return total

    def _metadata_path(self, data_path):
        root, _ = os.path.splitext(data_path)
        return f"{root}.meta.json"

    def _write_metadata(self, data_path, metadata):
        metadata = dict(metadata)
        metadata.setdefault("file_name", os.path.basename(data_path))
        metadata.setdefault("created_at", time.time())
        with open(self._metadata_path(data_path), "w") as f:
            json.dump(metadata, f, indent=2)

    def _write_wav(self, filepath, sample_rate, audio_data):
        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(audio_i16.tobytes())

    def _environment_class_counts(self):
        env_buffer_dir = os.path.join(self.base_dir, 'environment')
        if not os.path.exists(env_buffer_dir):
            return {}

        class_counts = {}
        for class_dir in os.listdir(env_buffer_dir):
            if self.environment_classes is not None and class_dir not in self.environment_classes:
                continue
            class_path = os.path.join(env_buffer_dir, class_dir)
            if os.path.isdir(class_path):
                count = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                if count > 0:
                    class_counts[class_dir] = count
        return class_counts

    def _top_environment_classes(self):
        class_counts = self._environment_class_counts()
        return sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:self.env_target_classes]

    def has_environment_top_classes(self):
        """Ready to choose dynamic top-N classes once N classes appeared at least once."""
        return len(self._top_environment_classes()) >= self.env_target_classes

    def get_status(self):
        """Return current counts per class per modality."""
        status = {}
        for modality, classes in self.thresholds.items():
            status[modality] = {}
            for class_name in classes.keys():
                status[modality][class_name] = self.get_count(modality, class_name)
        return status

    def is_emotion_phase_1_ready(self):
        """Check if all 5 automatic emotion classes >= 50 images."""
        auto_classes = ['angry', 'disgust', 'happy', 'neutral', 'surprise']
        for c in auto_classes:
            if self.get_count('emotion', c) < self.thresholds['emotion'][c]:
                return False
        return True

    def are_all_individual_models_ready(self):
        """Check if ALL individual models have met their per-class minimums."""
        # Excluding Health from retraining requirements
        for modality, classes in self.thresholds.items():
            if modality == 'health': 
                continue
            if modality == 'environment':
                if not self.is_modality_ready('environment'):
                    return False
                continue
            for class_name, required_count in classes.items():
                if self.get_count(modality, class_name) < required_count:
                    return False
        return True

    def is_modality_ready(self, modality):
        """Check if a specific modality has met all its per-class minimums."""
        # Special case: ENVIRONMENT is dynamic. 
        if modality == 'environment':
            # Check if others are ready to "force" environment
            others_ready = all(self.retrain_status[m] in ['running', 'completed'] for m in ['emotion', 'gesture', 'speech'])
            
            top_classes = self._top_environment_classes()
            # Normal trigger: we have top 5 and they are full
            if len(top_classes) >= self.env_target_classes:
                ready = all(count >= self.env_class_threshold for _, count in top_classes)
                if ready: return True

            # Forced trigger: others are ready AND we have at least one valid top class
            if others_ready and len(top_classes) > 0:
                print(f"⏩ [Continual Learning] Environment target not reached, but forcing upgrade to match other modalities.")
                return True
                
            return False
        
        # Standard logic for fixed modalities (emotion, gesture, speech, health)
        if modality not in self.thresholds:
            return False
            
        for class_name, required_count in self.thresholds[modality].items():
            if self.get_count(modality, class_name) < required_count:
                return False
        return True

    def check_and_trigger_retraining(self):
        """
        Check if any modality thresholds are met and launch their train script
        in the background if they haven't been started yet.
        """
        # 1. Update status of currently running processes
        for mod, proc in list(self.retrain_processes.items()):
            if proc.poll() is not None:
                log_handle = self.retrain_log_handles.pop(mod, None)
                if log_handle is not None:
                    log_handle.close()
                # Process finished
                if proc.returncode == 0:
                    print(f"\n[INFO] {mod.upper()} retraining completed successfully!")
                    self.retrain_status[mod] = 'completed'
                    
                    # Increment version and DOUBLE thresholds for the next cycle
                    if mod != 'fusion':
                        self._increment_version(mod)
                else:
                    print(f"\n[ERR] {mod.upper()} retraining failed (Exit Code {proc.returncode}). Check logs.")
                    self.retrain_status[mod] = 'failed'
                del self.retrain_processes[mod]

        # 2. Check each individual branch model
        individual_modalities = ['emotion', 'gesture', 'speech', 'environment']
        for mod in individual_modalities:
            if self.retrain_status[mod] == 'pending':
                # SPECIAL: for environment, select dynamic top 5 first, then augment
                # those top-5 buffer classes to the per-class threshold.
                if mod == 'environment' and self.has_environment_top_classes():
                    print(f"\n📸 [Augmentation] Running environment buffer augmentation before retraining...")
                    self.augment_environment_buffer()
                
                # SPECIAL: for emotion, augment Disgust if quota not met
                if mod == 'emotion' and self.get_count('emotion', 'disgust') < self.thresholds['emotion']['disgust']:
                    print(f"\n📸 [Augmentation] Running DISGUST augmentation before emotion retraining...")
                    self.augment_emotion_disgust()

                if not self.is_modality_ready(mod):
                    continue
                
                # Trigger training!
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.retrain_scripts[mod])
                print(f"\n🚀 [Continual Learning] Thresholds met for {mod.upper()}! Auto-launching {self.retrain_scripts[mod]} in background...")
                
                # Verify script exists to avoid crash
                if not os.path.exists(script_path):
                    print(f"[WARN] Training script {script_path} not found. Cannot launch.")
                    # Fallback to prevent spam
                    self.retrain_status[mod] = 'failed' 
                    continue

                # Launch subprocess (non-blocking), writing logs for debugging.
                logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "continual_learning")
                os.makedirs(logs_dir, exist_ok=True)
                log_path = os.path.join(logs_dir, f"{mod}_{int(time.time())}.log")
                log_handle = open(log_path, "w")
                print(f"[INFO] {mod.upper()} retraining log: {log_path}")
                proc = subprocess.Popen(
                    [sys.executable, script_path], 
                    stdout=log_handle,
                    stderr=subprocess.STDOUT
                )
                self.retrain_processes[mod] = proc
                self.retrain_log_handles[mod] = log_handle
                self.retrain_status[mod] = 'running'

        # 3. Check for Fusion (ONLY triggers when ALL individual models show 'completed')
        all_completed = all(self.retrain_status[mod] == 'completed' for mod in individual_modalities)
        if all_completed and self.retrain_status['fusion'] == 'pending':
            mod = 'fusion'
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.retrain_scripts[mod])
            print(f"\n🔥 [Continual Learning] ALL INDIVIDUAL MODELS UPDATED! Launching FUSION fine-tuning...")
            
            if os.path.exists(script_path):
                logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "continual_learning")
                os.makedirs(logs_dir, exist_ok=True)
                log_path = os.path.join(logs_dir, f"{mod}_{int(time.time())}.log")
                log_handle = open(log_path, "w")
                print(f"[INFO] FUSION retraining log: {log_path}")
                proc = subprocess.Popen([sys.executable, script_path], stdout=log_handle, stderr=subprocess.STDOUT)
                self.retrain_processes[mod] = proc
                self.retrain_log_handles[mod] = log_handle
                self.retrain_status[mod] = 'running'
            else:
                self.retrain_status[mod] = 'failed'

    def save_image_frame(self, frame, modality, class_name, confidence):
        """Save an image frame to the buffer if confidence exceeds 90%."""
        if confidence <= self.confidence_threshold:
            return False

        timestamp = time.time()
        metadata = {
            "modality": modality,
            "class_name": class_name,
            "confidence": float(confidence),
            "source": "live_inference",
            "augmented": False,
            "patient_specific": True,
            "created_at": timestamp,
        }

        # For ENVIRONMENT: Dynamic thresholds (no pre-defined class list)
        if modality == 'environment':
            if self.environment_classes is not None and class_name not in self.environment_classes:
                return False
            # Environment doesn't have a pre-defined threshold; create folders on-the-fly
            # Just create the directory if it doesn't exist and save
            filename = f"{modality}_patient_{class_name}_{int(timestamp * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(self.base_dir, modality, class_name, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if cv2.imwrite(filepath, frame):
                self._write_metadata(filepath, metadata)
                return True
            return False
        
        # For fixed modalities: Check thresholds
        current_count = self.get_count(modality, class_name)
        required_count = self.thresholds[modality].get(class_name, 0)
        
        if current_count >= required_count:
            return False # We already have enough for this class

        filename = f"{modality}_patient_{class_name}_{int(timestamp * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = os.path.join(self.base_dir, modality, class_name, filename)
        
        if cv2.imwrite(filepath, frame):
            self._write_metadata(filepath, metadata)
            return True
        return False

    def save_audio_sample(self, audio_data, sample_rate, class_name, confidence):
        """Save a speech audio snippet to buffer."""
        if confidence <= self.speech_conf_threshold:
            return False

        current_count = self.get_count('speech', class_name)
        required_count = self.thresholds['speech'].get(class_name, 0)
        
        if current_count >= required_count:
            return False

        filename = f"speech_patient_{class_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
        filepath = os.path.join(self.base_dir, 'speech', class_name, filename)
        
        self._write_wav(filepath, sample_rate, audio_data)
        self._write_metadata(filepath, {
            "modality": "speech",
            "class_name": class_name,
            "confidence": float(confidence),
            "sample_rate": int(sample_rate),
            "num_samples": int(len(audio_data)) if hasattr(audio_data, "__len__") else None,
            "source": "live_inference",
            "augmented": False,
            "patient_specific": True,
        })
        return True

    def save_health_signal(self, signal_data, class_name, confidence):
        """Save physiological signal data to buffer (NO augmentation)."""
        if confidence <= self.confidence_threshold:
            return False

        current_count = self.get_count('health', class_name)
        required_count = self.thresholds['health'].get(class_name, 0)
        
        if current_count >= required_count:
            return False

        filename = f"health_patient_{class_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.json"
        filepath = os.path.join(self.base_dir, 'health', class_name, filename)
        
        with open(filepath, 'w') as f:
            json.dump({"confidence": float(confidence), "data": signal_data.tolist() if isinstance(signal_data, np.ndarray) else signal_data}, f)
        self._write_metadata(filepath, {
            "modality": "health",
            "class_name": class_name,
            "confidence": float(confidence),
            "source": "live_inference",
            "augmented": False,
            "patient_specific": True,
        })
        
        return True

    def check_and_handle_manual_capture(self, cap, emotion_model, device, transform, get_class_label_func):
        """
        Check if we need to pause for manual capture (Fear/Sad).
        Shares the existing 'cap' (VideoCapture) object to avoid hardware conflicts.
        """
        # If phase 1 is done, or we already collected everything, skip
        if self.phase_1_complete or not self.is_emotion_phase_1_ready():
            return

        # Start manual capture
        print("\n" + "=" * 60)
        print("✓ EMOTION COLLECTION PHASE 1 COMPLETE (5/7 emotions ready)")
        print("=" * 60)
        print("\nStatus:")
        auto_classes = ['angry', 'happy', 'neutral', 'surprise', 'disgust']
        for c in auto_classes:
            print(f"  ✓ {c.capitalize()}:\t50/50 images")

        for manual_c in ['fear', 'sad']:
            if self.get_count('emotion', manual_c) < self.thresholds['emotion'][manual_c]:
                print(f"  ✗ {manual_c.capitalize()}:\t0/{self.thresholds['emotion'][manual_c]} images  (MANUAL CAPTURE NEEDED)")

        for manual_class in ['fear', 'sad']:
            needed = self.thresholds['emotion'][manual_class] - self.get_count('emotion', manual_class)
            if needed <= 0:
                continue
            
            print(f"\n📸 ENTER MANUAL CAPTURE MODE - {manual_class.upper()}")
            print(f"  - Make {manual_class} expressions to camera")
            print("  - System auto-saves frames where confidence > 90%")
            print(f"  - Collect until: buffers/emotion/{manual_class}/ has 50 images")
            print("  - Press SPACEBAR when complete or wait for auto-completion")
            input(f"\nPress ENTER to start {manual_class.upper()} manual capture...")

            # Use the existing capture object passed from main loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Predict emotion
                try:
                    from PIL import Image
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    x = transform(img).unsqueeze(0).to(device)
                    out = emotion_model(x)
                    lbl_name, lbl_conf = get_class_label_func(out, ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
                    
                    saved = False
                    if lbl_name == manual_class:
                        saved = self.save_image_frame(frame, 'emotion', lbl_name, lbl_conf)
                except Exception as e:
                    lbl_name, lbl_conf, saved = "error", 0.0, False
                    print(e)
                
                count = self.get_count('emotion', manual_class)
                display_frame = frame.copy()
                
                # Show status on screen
                status_text = f"Class: {manual_class.upper()} | Need: {count}/{self.thresholds['emotion'][manual_class]}"
                conf_text = f"Cur Pred: {lbl_name} ({lbl_conf:.1f}%) -> {'SAVED' if saved else 'Wait'}"
                
                color = (0, 255, 0) if saved else (0, 0, 255)
                cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.imshow(f"MANUAL CAPTURE: {manual_class.upper()}", display_frame)
                key = cv2.waitKey(1)
                
                if (key & 0xFF) == ord(' '):  # Spacebar to skip/finish early (fallback)
                    break
                
                if count >= self.thresholds['emotion'][manual_class]:
                    print(f"✓ Collected required images for {manual_class}!")
                    time.sleep(1) # Let user see they're done
                    break

            # DO NOT cap.release() here, it's the main camera!
            cv2.destroyWindow(f"MANUAL CAPTURE: {manual_class.upper()}")

        self.phase_1_complete = True
        print("\n=== MANUAL CAPTURE COMPLETE. ALL EMOTIONS READY ===")

    def augment_environment_buffer(self):
        """
        For Environment modality ONLY:
        1. Identify Top 5 classes from live buffer.
        2. Augment Top 5 classes from their own buffer images until each has 25 images.
        3. For the remaining ~142 classes, augment from the main Places365 dataset 
           until each has 25 images in the buffer.
        """
        import torchvision.transforms as transforms
        from PIL import Image
        import glob
        import random
        
        env_buffer_dir = os.path.join(self.base_dir, 'environment')
        main_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "places365_filtered", "train")
        
        if not os.path.exists(env_buffer_dir):
            os.makedirs(env_buffer_dir, exist_ok=True)
        
        # 1. Identify Top 5 and the rest
        top_classes_list = self._top_environment_classes()
        top_class_names = [c for c, _ in top_classes_list]
        
        all_env_classes = list(self.environment_classes) if self.environment_classes else []
        if not all_env_classes:
            print("[WARN] No environment classes defined. Skipping augmentation.")
            return

        print(f"\n[AUGMENT] Processing Environment augmentation (147 classes)...")
        
        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        
        total_augmented = 0
        target_count = 25

        for cls_name in all_env_classes:
            cls_path = os.path.join(env_buffer_dir, cls_name)
            os.makedirs(cls_path, exist_ok=True)
            
            orig_count = self.get_count('environment', cls_name)
            if orig_count >= target_count:
                continue

            need_to_augment = target_count - orig_count
            
            # Case A: Top 5 class -> Augment from buffer
            if cls_name in top_class_names:
                existing_images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not existing_images:
                    # Fallback to main dataset if buffer is empty for some reason
                    source_type = "main_dataset"
                    main_cls_path = os.path.join(main_dataset_dir, cls_name)
                    existing_images = glob.glob(os.path.join(main_cls_path, "*.jpg")) + \
                                      glob.glob(os.path.join(main_cls_path, "*.png"))
                else:
                    source_type = "live_buffer"
                    existing_images = [os.path.join(cls_path, f) for f in existing_images]
            
            # Case B: Other 142 classes -> Augment from main dataset
            else:
                source_type = "main_dataset"
                main_cls_path = os.path.join(main_dataset_dir, cls_name)
                existing_images = glob.glob(os.path.join(main_cls_path, "*.jpg")) + \
                                  glob.glob(os.path.join(main_cls_path, "*.png"))

            if not existing_images:
                # print(f"  [WARN] {cls_name}: No source images found in {source_type}.")
                continue
            
            # print(f"  ⚙ {cls_name}: Adding {need_to_augment} images from {source_type}")
            
            for aug_idx in range(need_to_augment):
                src_img_path = random.choice(existing_images)
                try:
                    img = Image.open(src_img_path).convert('RGB')
                    aug_img = augment_transform(img)
                    
                    aug_filename = f"{cls_name}_aug_{uuid.uuid4().hex[:8]}.jpg"
                    aug_filepath = os.path.join(cls_path, aug_filename)
                    aug_img.save(aug_filepath)
                    
                    self._write_metadata(aug_filepath, {
                        "modality": "environment",
                        "class_name": cls_name,
                        "source": source_type,
                        "source_file": os.path.basename(src_img_path),
                        "augmented": True,
                        "patient_specific": (source_type == "live_buffer"),
                        "augmentation": ["horizontal_flip", "rotation", "zoom", "brightness", "contrast"],
                    })
                    total_augmented += 1
                except Exception as e:
                    pass
        
        print(f"✓ Environment augmentation complete: {total_augmented} new images added across 147 classes.")
    def augment_emotion_disgust(self):
        """
        For Emotion modality: Augment 'disgust' class using RAF-DB dataset images.
        Required by plan: Flip, rotate (±15°), brightness (±0.2), contrast (±0.2).
        Source: datasets/raf-db-dataset/DATASET/train/3 (Index 1: Disgust)
        """
        import torchvision.transforms as transforms
        from PIL import Image
        import glob
        import random
        
        cls_name = 'disgust'
        cls_path = os.path.join(self.base_dir, 'emotion', cls_name)
        target_count = self.thresholds['emotion']['disgust']
        
        # RAF-DB mapping for Disgust is '3'
        raf_disgust_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "raf-db-dataset", "DATASET", "train", "3")
        
        orig_count = self.get_count('emotion', cls_name)
        if orig_count >= target_count:
            return

        print(f"⚙ Augmenting DISGUST from RAF-DB: {orig_count} → {target_count} images")
        
        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        
        # Get source images from RAF-DB
        raf_images = glob.glob(os.path.join(raf_disgust_dir, "*.jpg")) + glob.glob(os.path.join(raf_disgust_dir, "*.png"))
        
        if not raf_images:
            print(f"[WARN] No source images found in RAF-DB disgust folder: {raf_disgust_dir}")
            return

        total_augmented = 0
        need_to_augment = target_count - orig_count
        
        for _ in range(need_to_augment):
            src_img_path = random.choice(raf_images)
            try:
                img = Image.open(src_img_path).convert('RGB')
                aug_img = augment_transform(img)
                aug_filename = f"emotion_aug_disgust_rafdb_{uuid.uuid4().hex[:8]}.jpg"
                aug_filepath = os.path.join(cls_path, aug_filename)
                aug_img.save(aug_filepath)
                self._write_metadata(aug_filepath, {
                    "modality": "emotion",
                    "class_name": "disgust",
                    "source": "raf-db-dataset",
                    "source_file": os.path.basename(src_img_path),
                    "augmented": True,
                    "patient_specific": False, # From dataset
                    "augmentation": ["horizontal_flip", "rotation", "brightness", "contrast"],
                })
                total_augmented += 1
            except:
                pass
        print(f"✓ Disgust augmentation complete: {total_augmented} new images added from RAF-DB.")
    def _load_versions(self):
        """Load or initialize model versions and scaled thresholds."""
        # v1 Defaults (The "Step Size" for linear growth)
        self.v1_defaults = {
            'emotion': 50,
            'gesture': 40,
            'speech': 30,
            'environment': 25
        }
        
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                data = json.load(f)
                self.versions = data.get("versions", {})
                self.thresholds = data.get("thresholds", self.thresholds)
                self.env_class_threshold = data.get("env_class_threshold", 25)
        else:
            # Initialize with v1 defaults
            self.versions = {
                'emotion': 1, 'gesture': 1, 'speech': 1, 'environment': 1, 'fusion': 1
            }
            self._save_versions()

    def _save_versions(self):
        with open(self.version_file, 'w') as f:
            json.dump({
                "versions": self.versions,
                "thresholds": self.thresholds,
                "env_class_threshold": self.env_class_threshold
            }, f, indent=4)

    def _increment_version(self, modality):
        """Increment version and add one v1-sized 'step' to the threshold."""
        self.versions[modality] += 1
        print(f"\n🆙 [Continual Learning] {modality.upper()} upgraded to v{self.versions[modality]}!")
        
        # Linear Increment (v1=50 -> v2=100 -> v3=150)
        step = self.v1_defaults.get(modality, 50)
        
        if modality == 'environment':
            self.env_class_threshold += step
            print(f"📈 Next environment goal: {self.env_class_threshold} images per class (+{step})")
        else:
            for cls in self.thresholds[modality]:
                self.thresholds[modality][cls] += step
            print(f"📈 Next {modality} goal: {list(self.thresholds[modality].values())[0]} samples per class (+{step})")
        
        self.retrain_status[modality] = 'pending' # Ready for next cycle
        self._save_versions()

    def get_progress_summary(self):
        """Return a dictionary for the UI Dashboard."""
        summary = {}
        for mod in ['emotion', 'gesture', 'speech', 'environment']:
            version = self.versions.get(mod, 1)
            status = self.retrain_status.get(mod, 'pending')
            
            if mod == 'environment':
                top_classes = self._top_environment_classes()
                # Sum only the counts of the top 5 classes (or fewer if we don't have 5)
                current = sum(count for _, count in top_classes[:self.env_target_classes])
                total = self.env_class_threshold * self.env_target_classes
            else:
                current_counts = [self.get_count(mod, cls) for cls in self.thresholds[mod]]
                current = sum(current_counts)
                total = sum(self.thresholds[mod].values())
            
            summary[mod] = {
                "version": version,
                "current": current,
                "total": total,
                "status": status
            }
        summary["fusion"] = {"version": self.versions.get("fusion", 1), "status": self.retrain_status.get("fusion", "pending")}
        return summary
