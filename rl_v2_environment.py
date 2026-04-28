"""
RL v2 Environment: Dynamic Modality Gating for Multimodal Fusion
================================================================
MDP Design:
  State:  5 modality confidences + 5 modality predictions (normalized) +
          previous gating strategy (one-hot 10) = 20-dim vector
  Action: Discrete(10) — 10 predefined gating weight strategies
  Reward: +1.0 correct weighted prediction, -0.5 wrong,
          +0.3 bonus for correctly ignoring a faulty modality,
          -0.3 penalty for trusting a wrong modality
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 10 Gating Strategies (weights for [emotion, env, health, gesture, speech]) ──
GATING_STRATEGIES = np.array([
    [0.20, 0.20, 0.20, 0.20, 0.20],  # 0: Equal trust
    [0.50, 0.15, 0.10, 0.10, 0.15],  # 1: Emotion-focused
    [0.15, 0.50, 0.10, 0.10, 0.15],  # 2: Environment-focused
    [0.10, 0.10, 0.50, 0.15, 0.15],  # 3: Health-focused
    [0.10, 0.10, 0.10, 0.50, 0.20],  # 4: Gesture-focused
    [0.15, 0.10, 0.10, 0.15, 0.50],  # 5: Speech-focused
    [0.35, 0.10, 0.10, 0.10, 0.35],  # 6: Emotion + Speech (emergency)
    [0.25, 0.25, 0.25, 0.00, 0.25],  # 7: Mask gesture
    [0.25, 0.25, 0.25, 0.25, 0.00],  # 8: Mask speech
    [0.30, 0.30, 0.10, 0.00, 0.30],  # 9: Vision + Audio (mask gesture+health)
], dtype=np.float32)

STRATEGY_NAMES = [
    "Equal", "Emotion-Focus", "Env-Focus", "Health-Focus",
    "Gesture-Focus", "Speech-Focus", "Emo+Speech", "Mask-Gesture",
    "Mask-Speech", "Vision+Audio"
]

NUM_CLASSES = 8
NUM_MODALITIES = 5
NUM_STRATEGIES = len(GATING_STRATEGIES)
CLASS_NAMES = ['normal', 'needs_attn', 'call_nurse', 'emergency',
               'agitated', 'distress_calm', 'shock', 'uncoop']

# ── Scenario Profiles: (modality_accuracy, confidence_range) per class ──
SCENARIO_PROFILES = {
    0: {'emotion': (0.90, 0.70, 0.95), 'env': (0.85, 0.60, 0.90),
        'health': (0.90, 0.70, 0.95), 'gesture': (0.30, 0.20, 0.50),
        'speech': (0.40, 0.20, 0.50)},
    1: {'emotion': (0.70, 0.50, 0.80), 'env': (0.60, 0.40, 0.70),
        'health': (0.65, 0.45, 0.75), 'gesture': (0.35, 0.20, 0.50),
        'speech': (0.50, 0.30, 0.60)},
    2: {'emotion': (0.50, 0.30, 0.60), 'env': (0.55, 0.40, 0.65),
        'health': (0.50, 0.35, 0.65), 'gesture': (0.80, 0.65, 0.95),
        'speech': (0.75, 0.60, 0.85)},
    3: {'emotion': (0.65, 0.45, 0.80), 'env': (0.40, 0.25, 0.55),
        'health': (0.85, 0.70, 0.95), 'gesture': (0.55, 0.35, 0.70),
        'speech': (0.95, 0.85, 0.99)},
    4: {'emotion': (0.80, 0.60, 0.90), 'env': (0.50, 0.30, 0.65),
        'health': (0.60, 0.40, 0.75), 'gesture': (0.45, 0.25, 0.60),
        'speech': (0.55, 0.35, 0.70)},
    5: {'emotion': (0.85, 0.70, 0.95), 'env': (0.55, 0.35, 0.70),
        'health': (0.70, 0.50, 0.80), 'gesture': (0.30, 0.15, 0.45),
        'speech': (0.40, 0.25, 0.55)},
    6: {'emotion': (0.75, 0.55, 0.90), 'env': (0.45, 0.30, 0.60),
        'health': (0.80, 0.65, 0.90), 'gesture': (0.50, 0.30, 0.65),
        'speech': (0.70, 0.50, 0.85)},
    7: {'emotion': (0.60, 0.40, 0.75), 'env': (0.50, 0.30, 0.65),
        'health': (0.55, 0.35, 0.70), 'gesture': (0.70, 0.50, 0.85),
        'speech': (0.45, 0.30, 0.60)},
}


class ModalityGatingEnv(gym.Env):
    """
    Gymnasium Environment for training an RL agent to dynamically select
    the optimal modality gating strategy for multimodal patient fusion.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.action_space = spaces.Discrete(NUM_STRATEGIES)
        # State: 5 confidences + 5 predictions (normalized) + 10 prev action one-hot
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_MODALITIES * 2 + NUM_STRATEGIES,),
            dtype=np.float32
        )
        self.ground_truth = None
        self.prev_action = 0
        self.step_count = 0
        self.max_steps = 100

    def _sample_modality_outputs(self, gt_class):
        """Generate realistic confidence + prediction for each modality."""
        profile = SCENARIO_PROFILES[gt_class]
        confidences = np.zeros(NUM_MODALITIES, dtype=np.float32)
        predictions = np.zeros(NUM_MODALITIES, dtype=np.float32)
        mod_keys = ['emotion', 'env', 'health', 'gesture', 'speech']

        for i, key in enumerate(mod_keys):
            acc_prob, lo, hi = profile[key]
            conf = np.random.uniform(lo, hi)
            confidences[i] = conf
            if np.random.rand() < acc_prob:
                predictions[i] = gt_class / (NUM_CLASSES - 1)  # normalize
            else:
                wrong = np.random.choice([c for c in range(NUM_CLASSES) if c != gt_class])
                predictions[i] = wrong / (NUM_CLASSES - 1)
        return confidences, predictions

    def _get_weighted_prediction(self, action, confidences, predictions):
        """Apply gating weights and return the weighted class prediction."""
        weights = GATING_STRATEGIES[action]
        raw_preds = (predictions * (NUM_CLASSES - 1)).astype(int)
        votes = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i in range(NUM_MODALITIES):
            votes[raw_preds[i]] += weights[i] * confidences[i]
        return int(np.argmax(votes))

    def _build_state(self, confidences, predictions):
        prev_one_hot = np.zeros(NUM_STRATEGIES, dtype=np.float32)
        prev_one_hot[self.prev_action] = 1.0
        return np.concatenate([confidences, predictions, prev_one_hot])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ground_truth = np.random.randint(0, NUM_CLASSES)
        self.prev_action = 0
        self.step_count = 0
        confs, preds = self._sample_modality_outputs(self.ground_truth)
        return self._build_state(confs, preds), {}

    def step(self, action):
        self.step_count += 1
        confs, preds = self._sample_modality_outputs(self.ground_truth)
        final_pred = self._get_weighted_prediction(action, confs, preds)

        # ── Reward Engineering ──
        reward = 1.0 if final_pred == self.ground_truth else -0.5
        # Consistency bonus
        if action == self.prev_action:
            reward += 0.2
        # Critical miss penalty (emergency)
        if self.ground_truth == 3 and final_pred != 3:
            reward -= 1.0

        self.prev_action = action
        # Periodically switch ground truth to simulate changing patient state
        if self.step_count % 20 == 0:
            self.ground_truth = np.random.randint(0, NUM_CLASSES)

        terminated = self.step_count >= self.max_steps
        state = self._build_state(confs, preds)
        info = {"gt": self.ground_truth, "pred": final_pred,
                "correct": final_pred == self.ground_truth,
                "strategy": STRATEGY_NAMES[action]}
        return state, reward, terminated, False, info


def make_gating_env(window_size=5):
    return ModalityGatingEnv(window_size=window_size)
