"""
Quick Setup Script — Installs dependencies and trains the DQN model
Run this once to get everything ready
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"[SETUP] {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False


def main():
    print(f"\n{'='*70}")
    print("Double DQN Fusion Smoother — Quick Setup")
    print(f"{'='*70}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Step 1: Install dependencies
    if not run_command(
        f"{sys.executable} install_rl_dependencies.py",
        "Step 1: Installing RL dependencies"
    ):
        print("\n[ERROR] Failed to install dependencies. Please install manually:")
        print("  pip install gymnasium==0.27.0 stable-baselines3==2.0.0")
        return False
    
    # Step 2: Train DQN
    if not run_command(
        f"{sys.executable} train_dqn_smoother.py",
        "Step 2: Training Double DQN model (~5-10 minutes)"
    ):
        print("\n[ERROR] Training failed.")
        return False
    
    # Step 3: Verify model
    model_path = "dqn_smoother.zip"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n✓ Model saved: {model_path} ({size_mb:.2f} MB)")
    else:
        print(f"\n[WARNING] Model file not found: {model_path}")
    
    print(f"\n{'='*70}")
    print("✓ Setup Complete!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("  1. Run real-time inference:")
    print(f"     python3 realtime_fusion_8cls.py")
    print("\n  2. Evaluate the trained model:")
    print(f"     python3 train_dqn_smoother.py eval")
    print("\nFor detailed information, see: RL_SETUP_GUIDE.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
