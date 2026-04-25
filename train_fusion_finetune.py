import os
import subprocess
import sys
import json
import time

# ==============================================================
# PHASE 2 - FUSION FINE-TUNING WRAPPER
# Description: 
# 1. Runs generate_v2_embeddings.py to extract fresh patient pools.
# 2. Generates a temporary model_manifest_v2.json.
# 3. Calls train_fusion_v3.py to update the fusion layer.
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V2_EMB_DIR = os.path.join(BASE_DIR, "v2_embeddings")
MANIFEST_PATH = os.path.join(BASE_DIR, "v2_embeddings", "model_manifest_latest.json")

def get_next_fusion_version():
    import glob
    import re
    pattern = os.path.join(BASE_DIR, "best_fusion_model_8cls_v*.pt")
    files = glob.glob(pattern)
    if not files:
        return 2
    def extract_version(f):
        match = re.search(r'_v(\d+)', f)
        return int(match.group(1)) if match else 0
    files.sort(key=extract_version, reverse=True)
    return extract_version(files[0]) + 1

NEXT_VERSION = get_next_fusion_version()
OUTPUT_MODEL = os.path.join(BASE_DIR, f"best_fusion_model_8cls_v{NEXT_VERSION}.pth")

def generate_manifest():
    print("--- Generating v2 Model Manifest ---")
    manifest = {
        "emotion": os.path.join(V2_EMB_DIR, "emotion_v2_pool.pt"),
        "gesture": os.path.join(V2_EMB_DIR, "gesture_v2_pool.pt"),
        "speech": os.path.join(V2_EMB_DIR, "speech_v2_pool.pt"),
        "environment": os.path.join(V2_EMB_DIR, "environment_v2_pool.pt"),
        "health": os.path.join(BASE_DIR, "health_embeddings.pt"), # Fallback to original as no live health yet
        "label_maps": os.path.join(BASE_DIR, "uploads", "metadata", "label_maps.json") # Re-use original mapping logic
    }
    
    # Check if all files exist
    for key, path in manifest.items():
        if not os.path.exists(path):
            print(f"[WARN] Manifest item missing: {key} at {path}")
            
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"✓ Manifest saved to: {MANIFEST_PATH}")

def run_retraining():
    print("\n🚀 Starting PHASE 2: Fusion Continual Learning...")
    
    # 1. Generate Embeddings
    extract_script = os.path.join(BASE_DIR, "generate_v2_embeddings.py")
    print(f"\n[Step 1/3] Extracting v2 embeddings using {extract_script}...")
    subprocess.check_call([sys.executable, extract_script])
    
    # 2. Build Manifest
    print(f"\n[Step 2/3] Building manifest...")
    generate_manifest()
    
    # 3. Run Fusion Training
    train_script = os.path.join(BASE_DIR, "train_fusion_v3.py")
    print(f"\n[Step 3/3] Running {train_script}...")
    cmd = [
        sys.executable,
        train_script,
        "--manifest", MANIFEST_PATH,
        "--out", OUTPUT_MODEL,
        "--epochs", "10",
        "--batch_size", "64"
    ]
    subprocess.check_call(cmd)
    
    print("\n" + "="*50)
    print("🎉 PHASE 2 COMPLETE: Fusion Model v2 is ready!")
    print("="*50)

if __name__ == '__main__':
    try:
        run_retraining()
    except Exception as e:
        print(f"\n[FATAL] Fusion retraining failed: {e}")
        sys.exit(1)
