"""
Gymnasium Environment for Fusion Model Smoothing using Double DQN
Simulates temporal sequences of fusion probabilities with intentional flickering
and trains the agent to pick the most stable decision.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FusionSmootherEnv(gym.Env):
    """
    Custom Gymnasium environment for training a Double DQN agent to stabilize
    8-class fusion model predictions over time.
    
    State: Last 5 frames of fusion probabilities (8 classes each) = 40-dim vector
    Action: One of 8 class predictions
    Reward: +1 for correct, -0.5 for flickering, -1 for dangerous misclassification
    """
    
    def __init__(self, window_size=5, num_classes=8):
        super().__init__()
        self.window_size = window_size
        self.num_classes = num_classes
        
        # Action space: 8 discrete actions (class indices 0-7)
        self.action_space = spaces.Discrete(num_classes)
        
        # State space: Flattened sliding window of probabilities + current frame
        # window_size * num_classes (probabilities) + num_classes (current scores)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(window_size * num_classes + num_classes,), 
            dtype=np.float32
        )
        
        # Scenario tracking
        self.current_scenario = None
        self.ground_truth_action = None
        self.prev_action = None
        self.step_count = 0
        self.max_steps = 100
        
        # Probability history (sliding window)
        self.prob_history = []
        
    def _generate_scenario(self):
        """Generate a random scenario with flickering probabilities."""
        scenario_idx = np.random.randint(0, 8)
        return scenario_idx
    
    def _generate_noisy_probabilities(self, ground_truth_idx):
        """
        Generate probability distribution with:
        - High probability for ground truth
        - Random noise
        - Occasional flickers to other classes
        """
        probs = np.random.dirichlet(np.ones(self.num_classes)) * 0.1  # Start with small noise
        probs[ground_truth_idx] += 0.7  # Boost ground truth significantly
        
        # Occasionally introduce strong flicker to adjacent class
        if np.random.rand() < 0.15:  # 15% chance of flicker
            flicker_idx = (ground_truth_idx + np.random.randint(1, self.num_classes)) % self.num_classes
            probs[flicker_idx] += 0.3
        
        probs /= probs.sum()  # Normalize
        return probs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        self.ground_truth_action = self._generate_scenario()
        self.prob_history = []
        self.prev_action = None
        self.step_count = 0
        
        # Initialize with 'window_size' frames
        for _ in range(self.window_size):
            probs = self._generate_noisy_probabilities(self.ground_truth_action)
            self.prob_history.append(probs)
        
        state = self._get_state()
        info = {}
        
        return state, info
    
    def _get_state(self):
        """Build the state vector from the probability history and current frame."""
        # Flatten the sliding window
        window_flat = np.concatenate(self.prob_history).astype(np.float32)
        
        # Current frame (most recent)
        current_frame = self.prob_history[-1].copy()
        
        # Concatenate: [history_flattened, current_frame]
        state = np.concatenate([window_flat, current_frame]).astype(np.float32)
        
        return state
    
    def step(self, action):
        """Execute one step of the environment."""
        self.step_count += 1
        
        # Generate new noisy probabilities for this step
        new_probs = self._generate_noisy_probabilities(self.ground_truth_action)
        self.prob_history.append(new_probs)
        
        # Keep only the last 'window_size' frames
        if len(self.prob_history) > self.window_size:
            self.prob_history.pop(0)
        
        # Calculate reward
        reward = 0.0
        
        # Base reward: correct classification
        if action == self.ground_truth_action:
            reward += 1.0
        else:
            reward -= 0.1  # Small penalty for wrong class
        
        # Flicker penalty: penalize if action differs from previous action
        if self.prev_action is not None and action != self.prev_action:
            reward -= 0.5  # Flicker penalty
        
        # Dangerous misclassification penalty (simulate danger scenarios)
        # Assume class 7 is 'call_nurse' -- very important to not miss
        if self.ground_truth_action == 7 and action != 7:
            reward -= 1.0  # Severe penalty for missing critical event
        
        self.prev_action = action
        
        # Check termination
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Get next state
        state = self._get_state()
        info = {
            "ground_truth": self.ground_truth_action,
            "action": action,
            "correct": action == self.ground_truth_action
        }
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        """Optional: render the environment state."""
        if self.ground_truth_action is not None:
            print(f"Step: {self.step_count}, Ground Truth: {self.ground_truth_action}, "
                  f"Prev Action: {self.prev_action}")


def make_fusion_env(window_size=5, num_classes=8):
    """Factory function to create the environment."""
    return FusionSmootherEnv(window_size=window_size, num_classes=num_classes)
