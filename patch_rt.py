import re

file_path = "/home/btech02_06/emotion_project/realtime_fusion_8cls.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add import
if "from buffer_manager import BufferManager" not in content:
    content = content.replace("import os, sys, time, queue, warnings\n", "import os, sys, time, queue, warnings\nfrom buffer_manager import BufferManager\n")

# 2. Init buffer manager
init_code = """    # Initialize Continual Learning Buffer Manager
    buffer_mgr = BufferManager()
"""
if "buffer_mgr = BufferManager()" not in content:
    content = content.replace("    try:\n        while True:", init_code + "    try:\n        while True:")

# 3. Add to the inference loop the save functions
inference_code_search = "                speech_emb,  speech_lbl  = get_speech_embedding(last_audio)"
save_code = """                speech_emb,  speech_lbl  = get_speech_embedding(last_audio)

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
                        frame, emotion_model, device, emotion_transform, get_class_label)
                # ========================================"""
if "# === Continual Learning Data Collection ===" not in content:
    content = content.replace(inference_code_search, save_code)

with open(file_path, "w") as f:
    f.write(content)

print("Patch applied.")
