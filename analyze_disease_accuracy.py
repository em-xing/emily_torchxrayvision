#!/usr/bin/env python3
"""
Analyze disease-specific preservation and accuracy for the best configuration
Shows per-disease correlation between original and transformed predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

def analyze_disease_preservation():
    """Analyze disease preservation per disease category"""
    
    sample_file = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/run_000/samples/samples_epoch_15.npy'
    output_dir = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/visualizations'
    
    print(f"📊 Loading samples from: {sample_file}")
    data = np.load(sample_file, allow_pickle=True).item()
    
    orig_probs = data['original_disease_probs']
    trans_probs = data['transformed_disease_probs']
    true_labels = data['true_labels']
    dataset_names = data['dataset_names']
    
    disease_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    num_diseases = min(orig_probs.shape[1], len(disease_labels))
    print(f"   Number of samples: {len(orig_probs)}")
    print(f"   Number of diseases: {num_diseases}")
    
    # Calculate per-disease metrics
    results = []
    
    for disease_idx in range(num_diseases):
        disease_name = disease_labels[disease_idx] if disease_idx < len(disease_labels) else f"Disease_{disease_idx}"
        
        orig_pred = orig_probs[:, disease_idx]
        trans_pred = trans_probs[:, disease_idx]
        true_label = true_labels[:, disease_idx] if disease_idx < true_labels.shape[1] else np.full(len(orig_probs), np.nan)
        
        # Filter out NaN labels
        valid_mask = ~np.isnan(true_label)
        
        if valid_mask.sum() == 0:
            print(f"⚠️  {disease_name}: No valid labels")
            continue
        
        orig_pred_valid = orig_pred[valid_mask]
        trans_pred_valid = trans_pred[valid_mask]
        true_label_valid = true_label[valid_mask]
        
        # Calculate correlation between original and transformed predictions
        if len(orig_pred_valid) > 1:
            correlation, p_value = pearsonr(orig_pred_valid, trans_pred_valid)
        else:
            correlation = np.nan
            p_value = np.nan
        
        # Calculate mean absolute difference
        mean_diff = np.mean(np.abs(orig_pred_valid - trans_pred_valid))
        
        # Calculate prevalence (how many positive cases)
        prevalence = np.mean(true_label_valid)
        num_positive = np.sum(true_label_valid == 1.0)
        
        # Calculate original and transformed mean predictions
        orig_mean = np.mean(orig_pred_valid)
        trans_mean = np.mean(trans_pred_valid)
        
        results.append({
            'Disease': disease_name,
            'Correlation': correlation,
            'P-value': p_value,
            'Mean Diff': mean_diff,
            'Prevalence': prevalence,
            'Num Positive': int(num_positive),
            'Num Samples': int(valid_mask.sum()),
            'Orig Mean Pred': orig_mean,
            'Trans Mean Pred': trans_mean,
            'Pred Shift': trans_mean - orig_mean
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Correlation', ascending=False)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("📊 DISEASE-SPECIFIC PRESERVATION ANALYSIS")
    print("=" * 100)
    print(f"\n{'Disease':<30} {'Corr':>6} {'Mean Δ':>8} {'Samples':>8} {'Pos':>5} {'Orig':>6} {'Trans':>6} {'Shift':>7}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['Disease']:<30} {row['Correlation']:6.3f} {row['Mean Diff']:8.4f} "
              f"{row['Num Samples']:8d} {row['Num Positive']:5d} "
              f"{row['Orig Mean Pred']:6.3f} {row['Trans Mean Pred']:6.3f} "
              f"{row['Pred Shift']:+7.4f}")
    
    print("\n" + "=" * 100)
    print(f"Overall Statistics:")
    print(f"  Mean Correlation: {df['Correlation'].mean():.4f}")
    print(f"  Min Correlation:  {df['Correlation'].min():.4f} ({df.loc[df['Correlation'].idxmin(), 'Disease']})")
    print(f"  Max Correlation:  {df['Correlation'].max():.4f} ({df.loc[df['Correlation'].idxmax(), 'Disease']})")
    print(f"  Mean Abs Diff:    {df['Mean Diff'].mean():.4f}")
    print("=" * 100)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Disease-Specific Preservation Analysis (Best Config #1)', fontsize=14, fontweight='bold')
    
    # 1. Correlation by disease
    ax1 = axes[0, 0]
    colors = ['green' if c > 0.9 else 'orange' if c > 0.8 else 'red' for c in df['Correlation']]
    ax1.barh(df['Disease'], df['Correlation'], color=colors, alpha=0.7)
    ax1.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax1.axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax1.set_xlabel('Correlation (Original vs Transformed)', fontweight='bold')
    ax1.set_title('Disease Prediction Preservation', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0.5, 1.0])
    
    # 2. Mean absolute difference by disease
    ax2 = axes[0, 1]
    ax2.barh(df['Disease'], df['Mean Diff'], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Mean Absolute Difference', fontweight='bold')
    ax2.set_title('Average Prediction Change', fontweight='bold')
    
    # 3. Scatter: Prevalence vs Correlation
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['Prevalence'], df['Correlation'], 
                         s=df['Num Samples']*2, c=df['Mean Diff'], 
                         cmap='coolwarm', alpha=0.6, edgecolors='black')
    for idx, row in df.iterrows():
        if row['Correlation'] < 0.85 or row['Prevalence'] > 0.3:
            ax3.annotate(row['Disease'], (row['Prevalence'], row['Correlation']),
                        fontsize=7, alpha=0.7)
    ax3.set_xlabel('Disease Prevalence', fontweight='bold')
    ax3.set_ylabel('Correlation', fontweight='bold')
    ax3.set_title('Prevalence vs Preservation', fontweight='bold')
    ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Mean Diff')
    
    # 4. Prediction shift by disease
    ax4 = axes[1, 1]
    colors_shift = ['red' if s > 0 else 'blue' for s in df['Pred Shift']]
    ax4.barh(df['Disease'], df['Pred Shift'], color=colors_shift, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Prediction Shift (Trans - Orig)', fontweight='bold')
    ax4.set_title('Direction of Prediction Change', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = f"{output_dir}/disease_preservation_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {output_file}")
    plt.close()
    
    # Save CSV
    csv_file = f"{output_dir}/disease_preservation_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Saved CSV: {csv_file}")
    
    # Create detailed scatter plots for each disease
    print("\n📊 Creating per-disease scatter plots...")
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('Per-Disease: Original vs Transformed Predictions', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, disease_idx in enumerate(range(num_diseases)):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        disease_name = disease_labels[disease_idx] if disease_idx < len(disease_labels) else f"Disease_{disease_idx}"
        
        orig_pred = orig_probs[:, disease_idx]
        trans_pred = trans_probs[:, disease_idx]
        
        # Scatter plot
        ax.scatter(orig_pred, trans_pred, alpha=0.5, s=20)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect preservation')
        
        # Add correlation
        corr = df[df['Disease'] == disease_name]['Correlation'].values[0] if disease_name in df['Disease'].values else np.nan
        
        ax.set_xlabel('Original Prediction', fontsize=9)
        ax.set_ylabel('Transformed Prediction', fontsize=9)
        ax.set_title(f'{disease_name}\nCorr: {corr:.3f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(num_diseases, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output_file = f"{output_dir}/per_disease_scatter.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved scatter plots: {output_file}")
    plt.close()

if __name__ == "__main__":
    analyze_disease_preservation()
