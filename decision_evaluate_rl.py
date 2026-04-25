import numpy as np
from rl_inference import FusionSmoother
from ppo_inference import PPOSmoother
from a2c_inference import A2CSmoother

def run_scenario(name, probabilities_sequence, dqn, ppo, a2c):
    print(f"\n{'='*80}")
    print(f"SCENARIO: {name}")
    print(f"{'='*80}")
    
    dqn.reset()
    ppo.reset()
    a2c.reset()
    
    print(f"{'Frame':<8} | {'Input Max Prob':<15} | {'DQN Decision':<15} | {'PPO Decision':<15} | {'A2C Decision':<15}")
    print("-" * 80)
    
    classes = ['normal', 'needs_attn', 'call_nurse', 'emergency', 
               'agitated', 'distress_calm', 'shock', 'uncoop']
    
    for i, probs in enumerate(probabilities_sequence):
        # Normalize
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # Raw argmax
        raw_idx = np.argmax(probs)
        raw_str = f"{classes[raw_idx]} ({probs[raw_idx]:.2f})"
        
        # DQN
        dqn_action, dqn_conf = dqn.update(probs)
        dqn_str = f"{classes[dqn_action]} ({dqn_conf:.2f})"
        
        # PPO
        ppo_action, ppo_conf = ppo.update(probs)
        ppo_str = f"{classes[ppo_action]} ({ppo_conf:.2f})"
        
        # A2C
        a2c_action, a2c_conf = a2c.update(probs)
        a2c_str = f"{classes[a2c_action]} ({a2c_conf:.2f})"
        
        print(f"Frame {i:02d} | {raw_str:<15} | {dqn_str:<15} | {ppo_str:<15} | {a2c_str:<15}")


if __name__ == "__main__":
    print("Loading Agents...")
    dqn = FusionSmoother(model_path="dqn_smoother")
    ppo = PPOSmoother(model_path="ppo_smoother")
    a2c = A2CSmoother(model_path="a2c_smoother")
    
    # helper to make flat probabilities
    def make_probs(target_idx, conf, num_classes=8):
        p = np.ones(num_classes) * ((1.0 - conf) / (num_classes - 1))
        p[target_idx] = conf
        return p

    # SCENARIO 1: Sudden True Emergency (Reaction Time)
    # 5 frames normal, then permanently Emergency
    s1 = []
    for _ in range(5): s1.append(make_probs(0, 0.8)) # 0: normal
    for _ in range(5): s1.append(make_probs(3, 0.8)) # 3: emergency
    
    run_scenario("1. SUDDEN EMERGENCY (Testing Reaction Time to True Threat)", s1, dqn, ppo, a2c)

    # SCENARIO 2: Conflicting / Ambiguous Signals
    # 0: normal, 1: needs_attn. They tie at 40% each.
    s2 = []
    for _ in range(10):
        p = np.full(8, 0.05)
        p[0] = 0.40 # normal
        p[1] = 0.40 # needs_attn
        # Slight noise
        p = p + np.random.normal(0, 0.05, 8)
        p = np.clip(p, 0.01, 1.0)
        s2.append(p)
        
    run_scenario("2. AMBIGUOUS INPUT (40% Normal vs 40% Needs Attn)", s2, dqn, ppo, a2c)

    # SCENARIO 3: Single Frame Noise Spike
    # Normal continuously, but one frame heavily spikes to "Agitated"
    s3 = []
    for i in range(10):
        if i == 5:
            s3.append(make_probs(4, 0.95)) # 4: agitated spike
        else:
            s3.append(make_probs(0, 0.85)) # 0: normal
            
    run_scenario("3. SINGLE FRAME NOISE SPIKE (Testing Robustness)", s3, dqn, ppo, a2c)
