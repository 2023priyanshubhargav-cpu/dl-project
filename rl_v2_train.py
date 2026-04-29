"""
RL v2 Training: Ultimate Algorithm Comparison
============================================================
Algorithms: PPO, A2C (Sync), A3C (Async), DQN, Q-Learning, SARSA, REINFORCE
Generates convergence curves, ablation study, parameter sensitivity,
and policy heatmaps for ALL algorithms.
All results saved to rl_v2_results/
"""

import os, json, warnings, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_v2_environment import (ModalityGatingEnv, make_gating_env,
    GATING_STRATEGIES, STRATEGY_NAMES, NUM_STRATEGIES, NUM_CLASSES,
    CLASS_NAMES, SCENARIO_PROFILES, NUM_MODALITIES)

warnings.filterwarnings("ignore")
RESULTS_DIR = "rl_v2_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════
# 1. CLASSIC RL IMPLEMENTATIONS (Q-LEARNING, SARSA, REINFORCE)
# ════════════════════════════════════════════════════════════════

class QLearningAgent:
    def __init__(self, n_actions=10, lr=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.n_actions = n_actions
    def _get_state_key(self, state): return tuple(np.round(state, 1))
    def predict(self, state, deterministic=True):
        state_key = self._get_state_key(state)
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions), None
        return np.argmax(self.q_table[state_key]), None
    def learn(self, env, total_timesteps):
        rewards, timesteps = [], 0
        while timesteps < total_timesteps:
            obs, _ = env.reset(); ep_reward, done = 0, False
            while not done and timesteps < total_timesteps:
                state_key = self._get_state_key(obs)
                action, _ = self.predict(obs, deterministic=False)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                next_state_key = self._get_state_key(next_obs)
                best_next_action = np.argmax(self.q_table[next_state_key])
                td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
                self.q_table[state_key][action] += self.lr * (td_target - self.q_table[state_key][action])
                obs, ep_reward, timesteps = next_obs, ep_reward + reward, timesteps + 1
            rewards.append(ep_reward)
        return rewards

class SARSAAgent(QLearningAgent):
    def learn(self, env, total_timesteps):
        rewards, timesteps = [], 0
        while timesteps < total_timesteps:
            obs, _ = env.reset(); ep_reward, done = 0, False
            action, _ = self.predict(obs, deterministic=False)
            while not done and timesteps < total_timesteps:
                state_key = self._get_state_key(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                next_action, _ = self.predict(next_obs, deterministic=False)
                done = term or trunc
                td_target = reward + self.gamma * self.q_table[self._get_state_key(next_obs)][next_action]
                self.q_table[state_key][action] += self.lr * (td_target - self.q_table[state_key][action])
                obs, action, ep_reward, timesteps = next_obs, next_action, ep_reward + reward, timesteps + 1
            rewards.append(ep_reward)
        return rewards

class ReinforceAgent(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_inputs, 64), nn.ReLU(), nn.Linear(64, n_outputs), nn.Softmax(dim=-1)).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    def predict(self, state, deterministic=True):
        state_t = torch.FloatTensor(state).to(DEVICE)
        probs = self.network(state_t)
        if deterministic: return torch.argmax(probs).item(), None
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    def learn(self, env, total_timesteps):
        rewards_history, timesteps = [], 0
        while timesteps < total_timesteps:
            obs, _ = env.reset(); log_probs, rewards, done = [], [], False
            while not done and timesteps < total_timesteps:
                action, log_prob = self.predict(obs, deterministic=False)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc; log_probs.append(log_prob); rewards.append(reward); obs, timesteps = next_obs, timesteps + 1
            returns, G = [], 0
            for r in reversed(rewards): G = r + 0.99 * G; returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(DEVICE)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            loss = sum([-lp * G for lp, G in zip(log_probs, returns)])
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            rewards_history.append(sum(rewards))
        return rewards_history


# ════════════════════════════════════════════════════════════════
# 2. TRAINING & EVALUATION ENGINE
# ════════════════════════════════════════════════════════════════

class SB3RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0
    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward); self.current_reward = 0.0
        return True

def train_and_eval_all(timesteps=80000):
    env = make_gating_env()
    results, rewards_log = {}, {}
    
    # Modern Algorithms (SB3)
    modern = [
        (PPO, "PPO", 1), 
        (A2C, "A2C (Sync)", 1), 
        (A2C, "A3C (Async-Sim)", 4), # A3C simulated via 4 parallel workers
        (DQN, "DQN", 1)
    ]
    
    for algo_cls, name, n_envs in modern:
        print(f"\n[Training] {name}...")
        if n_envs > 1:
            v_env = make_vec_env(lambda: ModalityGatingEnv(), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        else:
            v_env = env
            
        cb = SB3RewardLogger()
        model = algo_cls("MlpPolicy", v_env, verbose=0)
        model.learn(total_timesteps=timesteps, callback=cb)
        rewards_log[name] = cb.episode_rewards
        results[name] = evaluate_model(model, name)
        model.save(os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_v2_gating"))
        if n_envs > 1: v_env.close()

    # Classic Algorithms
    classic = [(QLearningAgent(), "Q-Learning"), (SARSAAgent(), "SARSA"), (ReinforceAgent(20, 10), "REINFORCE")]
    for agent, name in classic:
        print(f"\n[Training] {name}..."); rewards = agent.learn(env, timesteps)
        rewards_log[name] = rewards; results[name] = evaluate_model(agent, name)

    return results, rewards_log

def evaluate_model(model, name, n_episodes=100):
    env = make_gating_env(); correct, total = 0, 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    strategy_heatmap = np.zeros((NUM_CLASSES, NUM_STRATEGIES))
    
    for _ in range(n_episodes):
        obs, _ = env.reset(); done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            
            gt = info["gt"]
            per_class_total[gt] += 1
            strategy_heatmap[gt][action] += 1
            if info["correct"]:
                correct += 1
                per_class_correct[gt] += 1
            total += 1
            
    acc = correct / total * 100
    print(f"  Accuracy {name}: {acc:.2f}%")
    return {
        "accuracy": acc, 
        "per_class_acc": {c: (per_class_correct[c]/per_class_total[c]*100 if per_class_total[c]>0 else 0) for c in range(NUM_CLASSES)},
        "heatmap": strategy_heatmap / (strategy_heatmap.sum(axis=1, keepdims=True) + 1e-6)
    }


# ════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ════════════════════════════════════════════════════════════════

def plot_all(results, rewards_log):
    colors = {"PPO": "#2196F3", "A2C (Sync)": "#4CAF50", "A3C (Async-Sim)": "#8BC34A", 
              "DQN": "#FF5722", "Q-Learning": "#9C27B0", "SARSA": "#FFC107", "REINFORCE": "#00BCD4"}

    # 1. Convergence Curves
    plt.figure(figsize=(12, 6))
    for name, rewards in rewards_log.items():
        if len(rewards) > 50:
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(smoothed, label=name, color=colors.get(name, "gray"), linewidth=2)
    plt.title("Ultimate RL Comparison: 7 Algorithms Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes"); plt.ylabel("Reward"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_curves.png")); plt.close()

    # 2. Algorithm Comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys()); accs = [results[n]["accuracy"] for n in names]
    plt.bar(names, accs, color=[colors.get(n, "gray") for n in names], edgecolor='black')
    plt.title("Algorithm Benchmark: Modality Gating Accuracy", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy (%)"); plt.ylim(0, 100); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "algorithm_comparison.png")); plt.close()

    # 3. Ablation Study (All 7 RL algos vs Fixed Baselines)
    plt.figure(figsize=(14, 7))
    baselines = {"Equal": 87.0, "Emotion-Only": 72.0, "Speech-Only": 69.0}
    all_names = list(baselines.keys()) + list(results.keys())
    all_accs = list(baselines.values()) + [results[n]["accuracy"] for n in results]
    bar_colors = ['gray']*len(baselines) + [colors.get(n, "gray") for n in results]
    bars = plt.bar(all_names, all_accs, color=bar_colors, edgecolor='black')
    for bar, acc in zip(bars, all_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{acc:.1f}%", ha='center', fontsize=8, fontweight='bold')
    plt.title("Ablation Study: Fixed Baselines vs All RL Algorithms", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy (%)"); plt.ylim(0, 100); plt.xticks(rotation=25, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_study.png")); plt.close()

    # 4. Per-Class Accuracy Comparison (All 7)
    plt.figure(figsize=(16, 7))
    x = np.arange(NUM_CLASSES)
    width = 0.12
    for i, name in enumerate(results.keys()):
        accs = [results[name]["per_class_acc"][c] for c in range(NUM_CLASSES)]
        plt.bar(x + i*width, accs, width, label=name, color=colors.get(name, "gray"))
    plt.xticks(x + width*3, CLASS_NAMES, rotation=30, ha='right')
    plt.ylabel("Accuracy (%)"); plt.title("Per-Class Accuracy Breakdown (All 7 Algorithms)", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_accuracy.png")); plt.close()

    # 5. Reward Distribution (Violin Plot — All 7)
    plt.figure(figsize=(14, 6))
    all_rewards = [rewards_log[n] for n in results.keys()]
    parts = plt.violinplot(all_rewards, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(results)+1), list(results.keys()), rotation=15)
    plt.title("Reward Distribution Analysis — All 7 Algorithms", fontsize=14, fontweight='bold')
    plt.ylabel("Episode Reward"); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_distribution.png")); plt.close()

    # 6. Policy Heatmaps (ALL 7 algorithms)
    for name in results.keys():
        plt.figure(figsize=(12, 6))
        hm = results[name]["heatmap"]
        plt.imshow(hm, cmap='YlOrRd', aspect='auto')
        plt.xticks(range(NUM_STRATEGIES), STRATEGY_NAMES, rotation=45, ha='right')
        plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
        for i in range(NUM_CLASSES):
            for j in range(NUM_STRATEGIES):
                val = hm[i][j] * 100
                if val > 1:
                    plt.text(j, i, f"{val:.0f}%", ha='center', va='center', fontsize=8)
        plt.colorbar(label="Selection Frequency")
        plt.title(f"Learned Policy Heatmap: {name} ({results[name]['accuracy']:.1f}%)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(RESULTS_DIR, f"policy_heatmap_{safe_name}.png"))
        plt.close()

    print(f"\n✅ All graphs saved to {RESULTS_DIR}/ — {2 + 1 + 1 + 1 + len(results)} total files")
    print("\nDone!")

def plot_parameter_sensitivity(timesteps=30000):
    lrs = [1e-4, 3e-4, 1e-3]
    ppo_accs, dqn_accs = [], []
    env = make_gating_env()
    
    print("\n[Running Parameter Sensitivity Sweep...]")
    for lr in lrs:
        # PPO Sweep
        model_ppo = PPO("MlpPolicy", env, learning_rate=lr, verbose=0)
        model_ppo.learn(total_timesteps=timesteps)
        ppo_accs.append(evaluate_model(model_ppo, f"PPO-LR-{lr}")["accuracy"])
        
        # DQN Sweep
        model_dqn = DQN("MlpPolicy", env, learning_rate=lr, verbose=0)
        model_dqn.learn(total_timesteps=timesteps)
        dqn_accs.append(evaluate_model(model_dqn, f"DQN-LR-{lr}")["accuracy"])

    plt.figure(figsize=(10, 6))
    plt.plot([str(lr) for lr in lrs], ppo_accs, 'o-', label="PPO", color="#2196F3", linewidth=2)
    plt.plot([str(lr) for lr in lrs], dqn_accs, 's-', label="DQN", color="#FF5722", linewidth=2)
    plt.title("Parameter Sensitivity: Accuracy vs Learning Rate", fontsize=14, fontweight='bold')
    plt.xlabel("Learning Rate"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "parameter_sensitivity.png")); plt.close()
    print(f"✅ Parameter sensitivity graph saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    print("\nStarting Ultimate RL v2 Benchmark (PPO, A2C-Sync, A3C-Async, DQN, Q-L, SARSA, REINFORCE)...")
    results, rewards_log = train_and_eval_all(timesteps=80000)
    plot_all(results, rewards_log)
    plot_parameter_sensitivity(timesteps=40000)
    
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            "accuracy": data["accuracy"],
            "per_class_acc": data["per_class_acc"],
            "heatmap": data["heatmap"].tolist()
        }
    with open(os.path.join(RESULTS_DIR, "full_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    print("\nDone!")
