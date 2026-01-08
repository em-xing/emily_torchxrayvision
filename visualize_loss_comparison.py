#!/usr/bin/env python
"""
Visualize the difference between Disease Consistency Loss and Disease AUC Loss
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def l1_loss(pred, target):
    """Old consistency loss"""
    return torch.abs(pred - target).mean()

def auc_loss_single_disease(predictions, labels):
    """New AUC loss for single disease"""
    probs = torch.sigmoid(predictions)
    
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_probs = probs[pos_mask]
    neg_probs = probs[neg_mask]
    
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return torch.tensor(0.0)
    
    # Pairwise ranking
    pos_expanded = pos_probs.unsqueeze(1)
    neg_expanded = neg_probs.unsqueeze(0)
    pairwise_diff = pos_expanded - neg_expanded
    pairwise_loss = torch.nn.functional.softplus(-pairwise_diff)
    
    return pairwise_loss.mean()

def visualize_loss_comparison():
    """Compare old vs new loss on different scenarios"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Disease Consistency Loss vs Disease AUC Loss Comparison', fontsize=16, fontweight='bold')
    
    scenarios = [
        {
            'name': 'Scenario 1: Both Good',
            'orig': torch.tensor([3.0, 3.0, -3.0, -3.0]),
            'trans': torch.tensor([2.5, 2.8, -2.5, -2.8]),
            'labels': torch.tensor([1.0, 1.0, 0.0, 0.0])
        },
        {
            'name': 'Scenario 2: Original Good, Transformed Bad',
            'orig': torch.tensor([3.0, 3.0, -3.0, -3.0]),
            'trans': torch.tensor([-2.0, -2.5, 2.0, 2.5]),
            'labels': torch.tensor([1.0, 1.0, 0.0, 0.0])
        },
        {
            'name': 'Scenario 3: Both Bad',
            'orig': torch.tensor([-2.0, -2.5, 2.0, 2.5]),
            'trans': torch.tensor([-1.8, -2.3, 1.8, 2.3]),
            'labels': torch.tensor([1.0, 1.0, 0.0, 0.0])
        }
    ]
    
    for idx, scenario in enumerate(scenarios):
        orig = scenario['orig']
        trans = scenario['trans']
        labels = scenario['labels']
        
        # Calculate losses
        consistency_loss = l1_loss(trans, orig).item()
        auc_loss_val = auc_loss_single_disease(trans, labels).item()
        
        # Calculate AUCs
        orig_probs = torch.sigmoid(orig).numpy()
        trans_probs = torch.sigmoid(trans).numpy()
        labels_np = labels.numpy()
        
        orig_auc = roc_auc_score(labels_np, orig_probs)
        trans_auc = roc_auc_score(labels_np, trans_probs)
        
        # Plot ROC curves
        ax = axes[0, idx]
        
        fpr_orig, tpr_orig, _ = roc_curve(labels_np, orig_probs)
        fpr_trans, tpr_trans, _ = roc_curve(labels_np, trans_probs)
        
        ax.plot(fpr_orig, tpr_orig, 'b-', label=f'Original (AUC={orig_auc:.2f})', linewidth=2)
        ax.plot(fpr_trans, tpr_trans, 'r--', label=f'Transformed (AUC={trans_auc:.2f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k:', label='Random (AUC=0.50)', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(scenario['name'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot predictions
        ax = axes[1, idx]
        
        x_pos = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, orig_probs, width, label='Original', alpha=0.7, color='blue')
        ax.bar(x_pos + width/2, trans_probs, width, label='Transformed', alpha=0.7, color='red')
        
        # Mark ground truth
        for i, label in enumerate(labels_np):
            if label == 1:
                ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
                ax.text(i, 0.95, '✓', ha='center', fontsize=16, color='green', fontweight='bold')
            else:
                ax.text(i, 0.05, '✗', ha='center', fontsize=16, color='red', fontweight='bold')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Predicted Probability')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add loss comparison text
        loss_text = f'L1 Consistency: {consistency_loss:.3f}\nAUC Loss: {auc_loss_val:.3f}'
        ax.text(0.02, 0.98, loss_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/lotterlab/emily_torchxrayvision/loss_comparison_visualization.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved visualization: loss_comparison_visualization.png")
    
    # Print summary
    print("\n" + "="*80)
    print("LOSS COMPARISON SUMMARY")
    print("="*80)
    
    for idx, scenario in enumerate(scenarios):
        orig = scenario['orig']
        trans = scenario['trans']
        labels = scenario['labels']
        
        consistency_loss = l1_loss(trans, orig).item()
        auc_loss_val = auc_loss_single_disease(trans, labels).item()
        
        orig_probs = torch.sigmoid(orig).numpy()
        trans_probs = torch.sigmoid(trans).numpy()
        labels_np = labels.numpy()
        
        orig_auc = roc_auc_score(labels_np, orig_probs)
        trans_auc = roc_auc_score(labels_np, trans_probs)
        
        print(f"\n{scenario['name']}:")
        print(f"  Original AUC:      {orig_auc:.3f}")
        print(f"  Transformed AUC:   {trans_auc:.3f}")
        print(f"  AUC Drop:          {orig_auc - trans_auc:.3f}")
        print(f"  ---")
        print(f"  L1 Consistency:    {consistency_loss:.3f} {'(LOW - similar predictions)' if consistency_loss < 0.5 else '(HIGH - different predictions)'}")
        print(f"  AUC Loss:          {auc_loss_val:.3f} {'(LOW - good ranking)' if auc_loss_val < 0.7 else '(HIGH - poor ranking)'}")
        print(f"  ---")
        
        if idx == 0:
            print(f"  ✅ Both losses agree: predictions are good and similar")
        elif idx == 1:
            print(f"  ⚠️  L1 loss is MISLEADING: predictions are similar but INVERTED!")
            print(f"     → L1 loss says 'good' (similar)")
            print(f"     → AUC loss correctly says 'bad' (poor ranking)")
            print(f"     → This is why AUC loss is better!")
        elif idx == 2:
            print(f"  ❌ Both losses agree: predictions are bad")
            print(f"     But L1 loss thinks they're 'consistent' (both wrong)")
    
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("L1 Consistency Loss can be LOW even when predictions are WRONG,")
    print("as long as original and transformed are similarly wrong!")
    print("\nAUC Loss directly measures ACCURACY, not just similarity.")
    print("This is why it's a better objective for disease preservation.")
    print("="*80)

if __name__ == "__main__":
    visualize_loss_comparison()
