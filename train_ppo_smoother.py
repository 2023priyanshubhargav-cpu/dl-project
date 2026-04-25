"""
PPO (Proximal Policy Optimization) Training for Fusion Model Smoothing
On-policy alternative to DQN for comparison
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os
from rl_environment import make_fusion_env


def train_ppo_smoother(
    total_timesteps=100000,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    n_epochs=10,
    clip_range=0.2,
    window_size=5,
    num_classes=8,
    model_path="ppo_smoother"
):
    """
    Train PPO agent for fusion output smoothing.
    
    PPO is an on-policy algorithm that often provides better stability
    than off-policy methods like DQN.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for policy
        batch_size: Batch size for PPO updates
        n_steps: Number of steps per rollout
        n_epochs: Number of epochs for policy optimization
        clip_range: Clipping range for PPO
        window_size: Temporal window size
        num_classes: Number of classes (8)
        model_path: Path to save model
    """
    
    print("[PPO Training] Initializing environment...")
    env = make_fusion_env(window_size=window_size, num_classes=num_classes)
    
    print("[PPO Training] Creating PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log=None
    )
    
    print(f"[PPO Training] Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    print(f"[PPO Training] Saving model to {model_path}...")
    model.save(model_path)
    
    print("[PPO Training] Training complete!")
    
    return model


def evaluate_model(model_path="ppo_smoother", num_eval_episodes=10, window_size=5, num_classes=8):
    """
    Evaluate the trained PPO model.
    
    Args:
        model_path: Path to trained model
        num_eval_episodes: Number of evaluation episodes
        window_size: Temporal window size
        num_classes: Number of classes
    """
    print(f"\n[Evaluation] Loading model from {model_path}...")
    model = PPO.load(model_path)
    
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
    
    model_path = "ppo_smoother"
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        if os.path.exists(f"{model_path}.zip"):
            evaluate_model(model_path, num_eval_episodes=10)
        else:
            print(f"Model not found at {model_path}.zip. Train first!")
    else:
        # Training mode
        print("="*60)
        print("PPO Training for Fusion Output Smoothing")
        print("="*60)
        
        train_ppo_smoother(
            total_timesteps=100000,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            clip_range=0.2,
            window_size=5,
            num_classes=8,
            model_path=model_path
        )
        
        # Evaluate after training
        print("\nRunning evaluation on trained model...")
        evaluate_model(model_path, num_eval_episodes=10)
