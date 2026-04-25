"""
Train a Double DQN agent to smooth 8-class fusion model predictions.
Uses offline simulation of flickering probabilities to train the agent.
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os
from rl_environment import make_fusion_env


def train_dqn_smoother(
    total_timesteps=100000,
    learning_rate=1e-3,
    batch_size=32,
    buffer_size=10000,
    exploration_fraction=0.1,
    window_size=5,
    num_classes=8,
    model_path="dqn_smoother"
):
    """
    Train the Double DQN agent for fusion output smoothing.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for the neural network
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        exploration_fraction: Fraction of training for epsilon-greedy exploration
        window_size: Size of the temporal sliding window (frames)
        num_classes: Number of classification classes (8)
        model_path: Path to save the trained model
    """
    
    print("[DQN Training] Initializing environment...")
    env = make_fusion_env(window_size=window_size, num_classes=num_classes)
    
    print("[DQN Training] Creating Double DQN agent...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=None
    )
    
    print(f"[DQN Training] Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    print(f"[DQN Training] Saving model to {model_path}...")
    model.save(model_path)
    
    print("[DQN Training] Training complete!")
    
    return model


def evaluate_model(model_path="dqn_smoother", num_eval_episodes=10, window_size=5, num_classes=8):
    """
    Evaluate the trained model on fresh episodes.
    
    Args:
        model_path: Path to the trained model
        num_eval_episodes: Number of episodes to evaluate
        window_size: Size of the temporal sliding window
        num_classes: Number of classification classes
    """
    print(f"\n[Evaluation] Loading model from {model_path}...")
    model = DQN.load(model_path)
    
    env = make_fusion_env(window_size=window_size, num_classes=num_classes)
    
    total_reward = 0
    total_correct = 0
    total_steps = 0
    
    for episode in range(num_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_correct = 0
        episode_steps = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if info["correct"]:
                episode_correct += 1
            episode_steps += 1
            done = terminated or truncated
        
        total_reward += episode_reward
        total_correct += episode_correct
        total_steps += episode_steps
        
        accuracy = (episode_correct / episode_steps) * 100
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Accuracy={accuracy:.1f}%")
    
    avg_reward = total_reward / num_eval_episodes
    avg_accuracy = (total_correct / total_steps) * 100
    
    print(f"\n[Evaluation Summary]")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Accuracy: {avg_accuracy:.1f}%")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    # Check if model already exists
    model_path = "dqn_smoother"
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        if os.path.exists(f"{model_path}.zip"):
            evaluate_model(model_path, num_eval_episodes=10)
        else:
            print(f"Model not found at {model_path}.zip. Train first!")
    else:
        # Training mode
        print("="*60)
        print("Double DQN Training for Fusion Output Smoothing")
        print("="*60)
        
        train_dqn_smoother(
            total_timesteps=100000,
            learning_rate=1e-3,
            batch_size=32,
            buffer_size=10000,
            exploration_fraction=0.1,
            window_size=5,
            num_classes=8,
            model_path=model_path
        )
        
        # Evaluate after training
        print("\nRunning evaluation on trained model...")
        evaluate_model(model_path, num_eval_episodes=10)
