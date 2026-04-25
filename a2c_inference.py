import numpy as np
from collections import deque
from stable_baselines3 import A2C
import os

class A2CSmoother:
    def __init__(self, model_path="a2c_smoother", window_size=5, num_classes=8):
        self.window_size = window_size
        self.num_classes = num_classes
        self.model_path = model_path
        
        if os.path.exists(f"{model_path}.zip"):
            print(f"[A2CSmoother] Loading trained model from {model_path}...")
            self.model = A2C.load(model_path, device="cpu")
            self.model_loaded = True
        else:
            print(f"[A2CSmoother] WARNING: Model not found at {model_path}.zip")
            self.model = None
            self.model_loaded = False
        
        self.prob_history = deque(maxlen=window_size)
        self.is_ready = False
    
    def update(self, probabilities):
        probs = np.array(probabilities, dtype=np.float32)
        probs = probs / (probs.sum() + 1e-6)
        
        self.prob_history.append(probs)
        
        if len(self.prob_history) < self.window_size:
            return int(np.argmax(probs)), float(probs.max())
        
        self.is_ready = True
        
        if not self.model_loaded:
            return int(np.argmax(probs)), float(probs.max())
        
        state = self._build_state()
        
        action, _ = self.model.predict(state, deterministic=True)
        action = int(action)
        
        confidence = float(probs[action])
        return action, confidence
    
    def _build_state(self):
        window_flat = np.concatenate(list(self.prob_history)).astype(np.float32)
        current_frame = self.prob_history[-1].copy().astype(np.float32)
        state = np.concatenate([window_flat, current_frame]).astype(np.float32)
        return state
    
    def reset(self):
        self.prob_history.clear()
        self.is_ready = False
    
    def get_status(self):
        return {
            "model_loaded": self.model_loaded,
            "is_ready": self.is_ready,
            "history_size": len(self.prob_history),
            "window_size": self.window_size
        }

_a2c_smoother_instance = None

def get_a2c_smoother(model_path="a2c_smoother", window_size=5, num_classes=8):
    global _a2c_smoother_instance
    if _a2c_smoother_instance is None:
        _a2c_smoother_instance = A2CSmoother(model_path, window_size, num_classes)
    return _a2c_smoother_instance
