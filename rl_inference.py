"""
Real-time integration of Double DQN smoother into the fusion pipeline.
This module handles loading the trained model and smoothing predictions in real-time.
"""

import numpy as np
from collections import deque
from stable_baselines3 import DQN
import os


class FusionSmoother:
    """
    Wrapper class for applying Double DQN smoothing to real-time fusion predictions.
    """
    
    def __init__(self, model_path="dqn_smoother", window_size=5, num_classes=8):
        """
        Initialize the smoother with a trained DQN model.
        
        Args:
            model_path: Path to the trained DQN model (.zip file)
            window_size: Size of the temporal sliding window
            num_classes: Number of classification classes (8)
        """
        self.window_size = window_size
        self.num_classes = num_classes
        self.model_path = model_path
        
        # Load the trained model
        if os.path.exists(f"{model_path}.zip"):
            print(f"[FusionSmoother] Loading trained model from {model_path}...")
            self.model = DQN.load(model_path)
            self.model_loaded = True
        else:
            print(f"[FusionSmoother] WARNING: Model not found at {model_path}.zip")
            print("[FusionSmoother] Running in bypass mode (no smoothing)")
            self.model = None
            self.model_loaded = False
        
        # Initialize probability history
        self.prob_history = deque(maxlen=window_size)
        self.is_ready = False
    
    def update(self, probabilities):
        """
        Add new frame of probabilities to the history.
        
        Args:
            probabilities: Array of shape (num_classes,) with class probabilities
        
        Returns:
            smoothed_prediction: The DQN-smoothed class prediction (int 0-7)
            confidence: The confidence score for this prediction (0-1)
        """
        # Ensure probabilities are normalized
        probs = np.array(probabilities, dtype=np.float32)
        probs = probs / (probs.sum() + 1e-6)
        
        # Add to history
        self.prob_history.append(probs)
        
        # Check if we have enough history to make a decision
        if len(self.prob_history) < self.window_size:
            # Not ready yet; return the argmax prediction
            return int(np.argmax(probs)), float(probs.max())
        
        self.is_ready = True
        
        # If model is not loaded, fall back to argmax
        if not self.model_loaded:
            return int(np.argmax(probs)), float(probs.max())
        
        # Build the state vector for DQN
        state = self._build_state()
        
        # Get prediction from DQN (deterministic)
        action, _ = self.model.predict(state, deterministic=True)
        action = int(action)
        
        # Confidence is the probability of the predicted class
        confidence = float(probs[action])
        
        return action, confidence
    
    def _build_state(self):
        """
        Build the state vector from the probability history.
        
        Returns:
            state: Flattened array of shape (window_size * num_classes + num_classes,)
        """
        # Flatten the sliding window
        window_flat = np.concatenate(list(self.prob_history)).astype(np.float32)
        
        # Current frame (most recent)
        current_frame = self.prob_history[-1].copy().astype(np.float32)
        
        # Concatenate: [history_flattened, current_frame]
        state = np.concatenate([window_flat, current_frame]).astype(np.float32)
        
        return state
    
    def reset(self):
        """Reset the smoother (clear history). Call at the start of a new session."""
        self.prob_history.clear()
        self.is_ready = False
    
    def get_status(self):
        """Return the current status of the smoother."""
        return {
            "model_loaded": self.model_loaded,
            "is_ready": self.is_ready,
            "history_size": len(self.prob_history),
            "window_size": self.window_size
        }


# Singleton instance for use in realtime_fusion_8cls.py
_smoother_instance = None


def get_smoother(model_path="dqn_smoother", window_size=5, num_classes=8):
    """
    Get or create the global FusionSmoother instance.
    
    Args:
        model_path: Path to the trained DQN model
        window_size: Size of the temporal sliding window
        num_classes: Number of classification classes
    
    Returns:
        FusionSmoother instance
    """
    global _smoother_instance
    
    if _smoother_instance is None:
        _smoother_instance = FusionSmoother(
            model_path=model_path,
            window_size=window_size,
            num_classes=num_classes
        )
    
    return _smoother_instance


def reset_smoother():
    """Reset the global smoother instance."""
    global _smoother_instance
    if _smoother_instance is not None:
        _smoother_instance.reset()
