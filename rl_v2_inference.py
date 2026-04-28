"""
RL v2 Inference: Dynamic Modality Gating for Real-Time Fusion
=============================================================
Loads the trained PPO/A2C/DQN gating agent and provides a simple
interface for realtime_fusion_8cls.py to dynamically weight modalities.
"""

import os
import numpy as np
from collections import deque
from stable_baselines3 import PPO, A2C, DQN
from rl_v2_environment import GATING_STRATEGIES, STRATEGY_NAMES, NUM_STRATEGIES, NUM_MODALITIES

RESULTS_DIR = "rl_v2_results"


class ModalityGatingSmoother:
    """
    Real-time wrapper for the trained RL modality gating agent.
    Returns optimal modality weights given current sensor confidences.
    """

    def __init__(self, algo="PPO"):
        self.algo = algo.upper()
        path = os.path.join(RESULTS_DIR, f"{self.algo.lower()}_v2_gating")
        loader = {"PPO": PPO, "A2C": A2C, "DQN": DQN}.get(self.algo, PPO)

        if os.path.exists(f"{path}.zip"):
            print(f"[RL-v2 Gating] Loading {self.algo} agent from {path}")
            self.model = loader.load(path)
            self.loaded = True
        else:
            print(f"[RL-v2 Gating] WARNING: {path}.zip not found. Using equal weights.")
            self.model = None
            self.loaded = False

        self.prev_action = 0

    def get_weights(self, confidences, predictions):
        """
        Given current modality confidences and predictions,
        return the optimal gating weights.

        Args:
            confidences: np.array of shape (5,) — confidence per modality
            predictions: np.array of shape (5,) — predicted class per modality (normalized 0-1)

        Returns:
            weights: np.array of shape (5,) — trust weight per modality
            strategy_name: str — name of the selected strategy
        """
        if not self.loaded:
            return GATING_STRATEGIES[0].copy(), "Equal (fallback)"

        # Build state
        prev_oh = np.zeros(NUM_STRATEGIES, dtype=np.float32)
        prev_oh[self.prev_action] = 1.0
        state = np.concatenate([
            np.array(confidences, dtype=np.float32),
            np.array(predictions, dtype=np.float32),
            prev_oh
        ])

        action, _ = self.model.predict(state, deterministic=True)
        action = int(action)
        self.prev_action = action
        return GATING_STRATEGIES[action].copy(), STRATEGY_NAMES[action]

    def reset(self):
        self.prev_action = 0

    def get_status(self):
        return {"loaded": self.loaded, "algo": self.algo,
                "last_strategy": STRATEGY_NAMES[self.prev_action]}


# ── Singleton ──
_gating_instance = None

def get_gating_smoother(algo="PPO"):
    global _gating_instance
    if _gating_instance is None:
        _gating_instance = ModalityGatingSmoother(algo=algo)
    return _gating_instance
