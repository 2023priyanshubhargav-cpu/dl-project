# Double DQN Fusion Smoother — Setup & Usage Guide

## Overview
This system fixes "flickering" in the 8-class real-time fusion predictions by adding a trained Double DQN agent that acts as a temporal decision-maker. Instead of picking the highest probability at each frame, the DQN looks at the last 5 frames and selects the most stable prediction.

## File Structure

```
rl_environment.py           ← Gymnasium environment (simulates flickering + noise)
train_dqn_smoother.py       ← DQN training pipeline (offline)
rl_inference.py             ← Real-time inference wrapper (integrates into realtime_fusion_8cls.py)
realtime_fusion_8cls.py     ← Updated with smoother integration
install_rl_dependencies.py  ← Install gymnasium + stable-baselines3
```

---

## Step 1: Install RL Dependencies

```bash
cd /home/btech02_06/emotion_project
python3 install_rl_dependencies.py
```

Or manually:
```bash
pip install gymnasium==0.27.0 stable-baselines3==2.0.0
```

---

## Step 2: Train the DQN Model (Offline)

This trains the agent on **simulated** data (no GPU training needed, ~5-10 minutes):

```bash
python3 train_dqn_smoother.py
```

**What happens:**
1. Generates 100,000 fake frames of noisy probability sequences (simulating real fusion output with flickering)
2. Trains Double DQN to pick stable decisions (+1 reward for correct, -0.5 for flickering, -1 for missing critical events)
3. Saves the trained model as `dqn_smoother.zip`
4. Runs evaluation on fresh episodes

**Output:**
- `dqn_smoother.zip` — The trained model (lightweight, ~500KB)

---

## Step 3: Evaluate the Trained Model

```bash
python3 train_dqn_smoother.py eval
```

This runs 10 episodes on fresh simulation data and shows:
- Average reward
- Average accuracy (% of correct decisions)

---

## Step 4: Use the Smoother in Real-time

Once `dqn_smoother.zip` exists, simply run:

```bash
python3 realtime_fusion_8cls.py
```

**What happens automatically:**
1. The script loads the trained DQN model
2. Maintains a 5-frame sliding window of fusion probabilities
3. At each inference step, passes the window to the DQN
4. DQN outputs the smoothed prediction
5. This smoothed action is displayed on screen

**Console output shows:**
```
[00123] Action=normal         Conf=92.3% | emotion: happy (88.1%) | env: kitchen (76.2%) | ...
```

---

## Architecture Summary

### State (Input to DQN)
```
[p₀, p₁, ..., p₇] × 5 frames + [p₀, p₁, ..., p₇] current
= 48-dimensional vector
```
Where `p₀...p₇` are the 8-class probabilities from the fusion model.

### Action (Output)
```
One of 8 discrete actions: {normal, needs_attention, call_nurse, emergency, ...}
```

### Reward Function
- **+1.0**: Correct prediction
- **-0.1**: Wrong class
- **-0.5**: Flicker (action changed from previous frame)
- **-1.0**: Missed critical event (e.g., didn't predict 'call_nurse' when needed)

### Training Strategy
- **Offline**: Train on simulated data, not on live camera/audio
- **Fast**: ~100,000 timesteps = ~5-10 minutes on CPU
- **Robust**: Model generalizes to any flickering pattern

---

## Troubleshooting

### Model not found: "dqn_smoother.zip"
→ Run `python3 train_dqn_smoother.py` first to train the model

### RL inference module not available
→ Make sure `rl_inference.py` is in the same directory as `realtime_fusion_8cls.py`

### Predictions still flickering
→ Train longer: Edit `train_dqn_smoother.py`, increase `total_timesteps` from 100000 → 200000

### GPU out of memory
→ The smoother runs on CPU. DQN training is very lightweight.

---

## Fine-tuning

Edit `train_dqn_smoother.py` to adjust:

```python
train_dqn_smoother(
    total_timesteps=100000,      # ← Increase for longer training
    learning_rate=1e-3,           # ← Lower = more stable, slower
    batch_size=32,                # ← Larger = faster convergence
    buffer_size=10000,            # ← Replay buffer size
    exploration_fraction=0.1,     # ← How long to explore randomly
    window_size=5,                # ← Temporal window (5-10 frames recommended)
    num_classes=8,                # ← Must match fusion model
)
```

---

## Expected Performance

After training:
- **Before Smoother**: Class changes every 1-2 frames (high flicker)
- **After Smoother**: Predictions stabilize; changes only when confidence threshold crossed

Example:
```
Frame 1: NORMAL        (92%)
Frame 2: NEEDS_ATTENTION (51%)   ← Flicker!
Frame 3: NORMAL        (89%)     ← Flicker back!
Frame 4: NORMAL        (87%)

After Smoother:
Frame 1: NORMAL        (92%)
Frame 2: NORMAL        (88%)     ← Stable!
Frame 3: NORMAL        (85%)     ← Stable!
Frame 4: NORMAL        (87%)
```

---

## Next Steps

1. Train: `python3 train_dqn_smoother.py`
2. Run real-time: `python3 realtime_fusion_8cls.py`
3. Observe: Check if flickering is gone
4. Adjust: If still flickering, increase training iterations or adjust reward function in `rl_environment.py`

Happy smoothing! 🚀
