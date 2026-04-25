import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import make_fusion_env

def train_a2c_smoother(timesteps=100000):
    print("="*60)
    print("A2C Training for Fusion Output Smoothing")
    print("="*60)
    
    print("[A2C Training] Initializing environment...")
    env = make_fusion_env(window_size=5, num_classes=8)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    print("[A2C Training] Creating A2C agent...")
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=None, device="cpu")
    
    print(f"[A2C Training] Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    model_name = "a2c_smoother"
    print(f"[A2C Training] Saving model to {model_name}...")
    model.save(model_name)
    print("[A2C Training] Training complete!\n")
    
    print("Running evaluation on trained model...")
    evaluate_model(model_name, env)

def evaluate_model(model_name, env, num_episodes=10):
    print(f"\n[Evaluation] Loading model from {model_name}...")
    model = A2C.load(model_name, env=env, device="cpu")
    
    total_rewards = []
    total_accuracies = []
    
    for ep in range(num_episodes):
        obs = env.reset()[0]
        ep_reward = 0
        correct_predictions = 0
        steps = 0
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward
            if info.get('is_correct', False):
                correct_predictions += 1
            steps += 1
            
            if done or truncated:
                break
                
        accuracy = correct_predictions / steps if steps > 0 else 0
        total_rewards.append(ep_reward)
        total_accuracies.append(accuracy)
        print(f"  Episode {ep+1}: Reward={ep_reward[-1] if isinstance(ep_reward, list) else ep_reward:.2f}, Accuracy={accuracy*100:.1f}%")
        
    print("\n[Evaluation Summary]")
    print(f"  Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"  Avg Accuracy: {sum(total_accuracies)/len(total_accuracies)*100:.1f}%")

if __name__ == "__main__":
    train_a2c_smoother(timesteps=100000)
