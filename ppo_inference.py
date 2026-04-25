"""
PPO Real-time Inference Wrapper
Compatible with realtime_fusion_8cls.py for easy comparison with DQN
"""

import numpy as np
from collections import deque
from stable_baselines3 import PPO
import os


class PPOSmoother:
    """
    Wrapper for applying PPO smoothing to real-time fusion predictions.
    Interface is identical to FusionSmoother (DQN) for easy comparison.
    """
    
    def __init__(self, model_path="ppo_smoother", window_size=5, num_classes=8):
        """
        Initialize the PPO smoother.
        
        Args:
            model_path: Path to trained PPO model
            window_size: Temporal sliding window size
            num_classes: Number of classification classes (8)
        """
        self.window_size = window_size
        self.num_classes = num_classes
        self.model_path = model_path
        
        # Load the trained model
        if os.path.exists(f"{model_path}.zip"):
            print(f"[PPOSmoother] Loading trained model from {model_path}...")
            self.model = PPO.load(model_path)
            self.model_loaded = True
        else:
            print(f"[PPOSmoother] WARNING: Model not found at {model_path}.zip")
            print("[PPOSmoother] Running in bypass mode (no smoothing)")
            self.model = None
            self.model_loaded = False
        
        # Initialize probability history
        self.prob_history = deque(maxlen=window_size)
        self.is_ready = False
    
    def update(self, probabilities):
        """
        Add new frame of probabilities and return smoothed prediction.
        
        Args:
            probabilities: Array of shape (num_classes,) with class probabilities
        
        Returns:
            smoothed_prediction: The PPO-smoothed class prediction (int 0-7)
            confidence: Confidence score for this prediction (0-1)
        """
        # Normalize probabilities
        probs = np.array(probabilities, dtype=np.float32)
        probs = probs / (probs.sum() + 1e-6)
        
        # Add to history
        self.prob_history.append(probs)
        
        # Check if we have enough history
        if len(self.prob_history) < self.window_size:
            # Not ready; return argmax
            return int(np.argmax(probs)), float(probs.max())
        
        self.is_ready = True
        
        # If model not loaded, fall back to argmax
        if not self.model_loaded:
            return int(np.argmax(probs)), float(probs.max())
        
        # Build state vector for PPO
        state = self._build_state()
        
        # Get PPO prediction (deterministic)
        action, _ = self.model.predict(state, deterministic=True)
        action = int(action)
        
        # Confidence is the probability of predicted class
        confidence = float(probs[action])
        
        return action, confidence
    
    def _build_state(self):
        """
        Build state vector from probability history.
        
        Returns:
            state: Flattened array of shape (window_size * num_classes + num_classes,)
        """
        # Flatten sliding window
        window_flat = np.concatenate(list(self.prob_history)).astype(np.float32)
        
        # Current frame
        current_frame = self.prob_history[-1].copy().astype(np.float32)
        
        # Concatenate
        state = np.concatenate([window_flat, current_frame]).astype(np.float32)
        
        return state
    
    def reset(self):
        """Reset the smoother."""
        self.prob_history.clear()
        self.is_ready = False
    
    def get_status(self):
        """Return smoother status."""
        return {
            "model_loaded": self.model_loaded,
            "is_ready": self.is_ready,
            "history_size": len(self.prob_history),
            "window_size": self.window_size
        }


# Singleton instance
_ppo_smoother_instance = None


def get_ppo_smoother(model_path="ppo_smoother", window_size=5, num_classes=8):
    """
    Get or create the global PPO smoother instance.
    
    Args:
        model_path: Path to trained model
        window_size: Temporal window size
        num_classes: Number of classes
    
    Returns:
        PPOSmoother instance
    """
    global _ppo_smoother_instance
    
    if _ppo_smoother_instance is None:
        _ppo_smoother_instance = PPOSmoother(
            model_path=model_path,
            window_size=window_size,
            num_classes=num_classes
        )
    
    return _ppo_smoother_instance


def reset_ppo_smoother():
    """Reset the global PPO smoother."""
    global _ppo_smoother_instance
    if _ppo_smoother_instance is not None:
        _ppo_smoother_instance.reset()
