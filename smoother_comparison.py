"""
Comparison utilities for DQN vs PPO smoothers
Allows easy switching and performance metrics
"""

import numpy as np
from collections import defaultdict
import json
import time


class SmootherComparison:
    """
    Compare DQN and PPO smoothers side-by-side
    """
    
    def __init__(self, dqn_smoother=None, ppo_smoother=None):
        """
        Initialize comparison tracker.
        
        Args:
            dqn_smoother: DQN smoother instance
            ppo_smoother: PPO smoother instance
        """
        self.dqn_smoother = dqn_smoother
        self.ppo_smoother = ppo_smoother
        
        self.stats = {
            'dqn': defaultdict(list),
            'ppo': defaultdict(list),
        }
        
        self.frame_count = 0
        self.last_dqn_action = None
        self.last_ppo_action = None
    
    def update(self, probabilities, ground_truth=None):
        """
        Update both smoothers and collect stats.
        
        Args:
            probabilities: Current frame probabilities
            ground_truth: Optional ground truth label for accuracy calc
        
        Returns:
            results: Dict with DQN and PPO results
        """
        self.frame_count += 1
        
        results = {
            'frame': self.frame_count,
            'ground_truth': ground_truth,
            'dqn': None,
            'ppo': None,
        }
        
        # Get DQN prediction
        if self.dqn_smoother:
            dqn_action, dqn_conf = self.dqn_smoother.update(probabilities)
            results['dqn'] = {
                'action': dqn_action,
                'confidence': dqn_conf,
                'flicker': 1 if (self.last_dqn_action is not None and 
                                self.last_dqn_action != dqn_action) else 0,
            }
            self.stats['dqn']['actions'].append(dqn_action)
            self.stats['dqn']['confidences'].append(dqn_conf)
            self.stats['dqn']['flickers'].append(results['dqn']['flicker'])
            if ground_truth is not None:
                self.stats['dqn']['accuracy'].append(1 if dqn_action == ground_truth else 0)
            self.last_dqn_action = dqn_action
        
        # Get PPO prediction
        if self.ppo_smoother:
            ppo_action, ppo_conf = self.ppo_smoother.update(probabilities)
            results['ppo'] = {
                'action': ppo_action,
                'confidence': ppo_conf,
                'flicker': 1 if (self.last_ppo_action is not None and 
                                self.last_ppo_action != ppo_action) else 0,
            }
            self.stats['ppo']['actions'].append(ppo_action)
            self.stats['ppo']['confidences'].append(ppo_conf)
            self.stats['ppo']['flickers'].append(results['ppo']['flicker'])
            if ground_truth is not None:
                self.stats['ppo']['accuracy'].append(1 if ppo_action == ground_truth else 0)
            self.last_ppo_action = ppo_action
        
        return results
    
    def get_summary(self):
        """Get comparison summary statistics."""
        summary = {}
        
        for model in ['dqn', 'ppo']:
            if not self.stats[model]['actions']:
                continue
            
            summary[model] = {
                'frames_processed': len(self.stats[model]['actions']),
                'avg_confidence': np.mean(self.stats[model]['confidences']),
                'flicker_rate': np.mean(self.stats[model]['flickers']),
                'flicker_count': sum(self.stats[model]['flickers']),
            }
            
            if self.stats[model]['accuracy']:
                summary[model]['accuracy'] = np.mean(self.stats[model]['accuracy'])
        
        return summary
    
    def print_summary(self):
        """Print formatted comparison summary."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("SMOOTHER COMPARISON SUMMARY")
        print("="*70)
        
        for model in ['dqn', 'ppo']:
            if model not in summary:
                print(f"\n[{model.upper()}] Not initialized")
                continue
            
            stats = summary[model]
            print(f"\n[{model.upper()}]")
            print(f"  Frames Processed:    {stats['frames_processed']}")
            print(f"  Avg Confidence:      {stats['avg_confidence']:.3f}")
            print(f"  Flicker Rate:        {stats['flicker_rate']:.1%}")
            print(f"  Total Flickers:      {stats['flicker_count']}")
            
            if 'accuracy' in stats:
                print(f"  Accuracy:            {stats['accuracy']:.1%}")
        
        # Comparison
        if 'dqn' in summary and 'ppo' in summary:
            print("\n" + "-"*70)
            print("COMPARISON (PPO vs DQN)")
            print("-"*70)
            
            dqn_flicker = summary['dqn']['flicker_rate']
            ppo_flicker = summary['ppo']['flicker_rate']
            
            if ppo_flicker < dqn_flicker:
                print(f"✓ PPO has less flickering: {ppo_flicker:.1%} vs {dqn_flicker:.1%}")
            else:
                print(f"✓ DQN has less flickering: {dqn_flicker:.1%} vs {ppo_flicker:.1%}")
            
            dqn_conf = summary['dqn']['avg_confidence']
            ppo_conf = summary['ppo']['avg_confidence']
            
            if ppo_conf > dqn_conf:
                print(f"✓ PPO has higher confidence: {ppo_conf:.3f} vs {dqn_conf:.3f}")
            else:
                print(f"✓ DQN has higher confidence: {dqn_conf:.3f} vs {ppo_conf:.3f}")
        
        print("="*70 + "\n")
    
    def save_comparison(self, filename="comparison_results.json"):
        """Save comparison results to file."""
        summary = self.get_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Comparison] Results saved to {filename}")


def compare_smoothers_on_data(dqn_smoother, ppo_smoother, probabilities_list, ground_truths=None):
    """
    Compare two smoothers on a batch of probability sequences.
    
    Args:
        dqn_smoother: DQN smoother instance
        ppo_smoother: PPO smoother instance
        probabilities_list: List of probability arrays
        ground_truths: Optional list of ground truth labels
    
    Returns:
        comparison: SmootherComparison object with results
    """
    comparison = SmootherComparison(dqn_smoother, ppo_smoother)
    
    for i, probs in enumerate(probabilities_list):
        gt = ground_truths[i] if ground_truths else None
        comparison.update(probs, ground_truth=gt)
    
    return comparison

if __name__ == "__main__":
    from rl_inference import FusionSmoother
    from ppo_inference import PPOSmoother
    
    print("Testing DQN vs PPO smoothers with simulated noisy data...")
    dqn = FusionSmoother(model_path="dqn_smoother")
    ppo = PPOSmoother(model_path="ppo_smoother")
    
    comparison = set()
    comparison_obj = SmootherComparison(dqn_smoother=dqn, ppo_smoother=ppo)
    
    # Generate noisy sequence exactly like in training
    num_steps = 150
    ground_truth = []
    probabilities = []
    
    current_label = 0
    steps_since_change = 0
    change_interval = 30
    
    for _ in range(num_steps):
        steps_since_change += 1
        if steps_since_change >= change_interval:
            current_label = (current_label + 1) % 8
            steps_since_change = 0
            
        ground_truth.append(current_label)
        
        # Base probabilities
        probs = np.full(8, 0.05)
        probs[current_label] = 0.65
        
        # Add noise
        noise = np.random.normal(0, 0.15, 8)
        probs = probs + noise
        probs = np.clip(probs, 0.05, 1.0)
        
        # Simulate flickering (30% chance wrong class spikes)
        if np.random.random() < 0.30:
            random_class = np.random.randint(0, 8)
            if random_class != current_label:
                probs[random_class] += 0.8
                
        probs = probs / probs.sum()
        probabilities.append(probs)
        
    print(f"Generated {num_steps} frames of test data. Running smoothers...")
    
    for probs, gt in zip(probabilities, ground_truth):
        comparison_obj.update(probs, ground_truth=gt)
        
    comparison_obj.print_summary()
