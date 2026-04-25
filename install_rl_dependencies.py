"""
Installation guide for RL dependencies.
Run this ONCE before training the DQN model.
"""

import subprocess
import sys

def install_rl_packages():
    """Install required RL packages."""
    packages = [
        "gymnasium==0.27.0",
        "stable-baselines3==2.0.0",
    ]
    
    print("[Setup] Installing RL packages...")
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("[Setup] All RL packages installed successfully!")


if __name__ == "__main__":
    install_rl_packages()
