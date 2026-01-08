#!/usr/bin/env python
"""
Visualize AutoStainer Results: Morphology Preservation Analysis
Epoch 25 samples from morphology_autostainer_20251204_153438
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import os

def visualize_autostainer_samples(sample_path, output_dir=None):
    """Comprehensive visualization of AutoStainer samples"""
    
    # Load sample data
    print(f"📊 Loading samples from: {sample_path}")
    sample_data = np.load(sample_path, allow_pickle=True).item()
    
    # Extract data
    original_images = sample_data['original_images']
    transformed_images = sample_data['transformed_images']
    transform_params = sample_data['transform_params']
    dataset_names = sample_data['dataset_names']
    orig_disease_probs = sample_data['original_disease_probs']
    trans_disease_probs = sample_data['transformed_disease_probs']
    true_labels = sample_data['true_labels']
    
    num_samples = original_images.shape[0]
    print(f"✅ Loaded {num_samples} samples")
    
    # Create output directory for visualizations
    if output_dir is None:
        output_dir = os.path.dirname(sample_path).replace('samples', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SAMPLE COMPARISON GRID
    print("🖼️ Creating sample comparison grid...")
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    fig.suptitle('AutoStainer: Original vs Transformed vs Difference (Epoch 25)', fontsize=16)
    
    for i in range(8):
        if i < num_samples:
            # Original image
            axes[0, i].imshow(original_images[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'{dataset_names[i]} Original', fontsize=10)
            axes[0, i].axis('off')
            
            # Transformed image
            axes[1, i].imshow(transformed_images[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title('Transformed', fontsize=10)
            axes[1, i].axis('off')
            
            # Difference map
            diff = np.abs(transformed_images[i, 0] - original_images[i, 0])
            im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
            axes[2, i].set_title(f'Diff (max={diff.max():.3f})', fontsize=10)
            axes[2, i].axis('off')
            
            # Add colorbar for difference
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'sample_comparison_grid.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {comparison_path}")
    
    # 2. TRANSFORMATION PARAMETERS ANALYSIS
    print("📊 Analyzing transformation parameters...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Transformation Parameters Analysis (Epoch 25)', fontsize=16)
    
    param_names = ['Brightness', 'Contrast', 'Gamma', 'Saturation']
    param_colors = ['blue', 'red', 'green', 'orange']
    
    for i, (param_name, color) in enumerate(zip(param_names, param_colors)):
        row = i // 2
        col = i % 2
        
        # Histogram of parameter values
        axes[row, col].hist(transform_params[:, i], bins=20, alpha=0.7, color=color, 
                           edgecolor='black', linewidth=0.5)
        axes[row, col].set_title(f'{param_name} Distribution')
        axes[row, col].set_xlabel('Parameter Value')
        axes[row, col].set_ylabel('Count')
        axes[row, col].axvline(x=1.0 if param_name in ['Contrast', 'Gamma', 'Saturation'] else 0.0, 
                              color='red', linestyle='--', alpha=0.7, label='Identity')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    params_path = os.path.join(output_dir, 'transformation_parameters.png')
    plt.savefig(params_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {params_path}")
    
    # 3. DISEASE PRESERVATION ANALYSIS
    print("🏥 Analyzing disease preservation...")
    
    # Calculate correlations for each disease
    disease_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
        'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    correlations = []
    aucs_original = []
    aucs_transformed = []
    
    for i in range(14):
        # Skip diseases with no variation in true labels
        if len(np.unique(true_labels[:, i])) < 2:
            correlations.append(np.nan)
            aucs_original.append(np.nan)
            aucs_transformed.append(np.nan)
            continue
            
        # Correlation between original and transformed predictions
        valid_mask = ~(np.isnan(orig_disease_probs[:, i]) | np.isnan(trans_disease_probs[:, i]))
        if valid_mask.sum() > 1:
            corr = np.corrcoef(orig_disease_probs[valid_mask, i], 
                              trans_disease_probs[valid_mask, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
            
            # AUC scores
            try:
                auc_orig = roc_auc_score(true_labels[valid_mask, i], orig_disease_probs[valid_mask, i])
                auc_trans = roc_auc_score(true_labels[valid_mask, i], trans_disease_probs[valid_mask, i])
                aucs_original.append(auc_orig)
                aucs_transformed.append(auc_trans)
            except:
                aucs_original.append(np.nan)
                aucs_transformed.append(np.nan)
        else:
            correlations.append(0)
            aucs_original.append(np.nan)
            aucs_transformed.append(np.nan)
    
    # Plot disease preservation
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Disease Preservation Analysis (Epoch 25)', fontsize=16)
    
    # Correlation plot
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    axes[0, 0].bar(range(len(correlations)), correlations, alpha=0.7)
    axes[0, 0].set_title(f'Disease Prediction Correlation (Avg: {np.nanmean(correlations):.3f})')
    axes[0, 0].set_xlabel('Disease Index')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_xticks(range(len(disease_labels)))
    axes[0, 0].set_xticklabels(disease_labels, rotation=45, ha='right')
    axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC comparison
    valid_indices = [i for i in range(14) if not (np.isnan(aucs_original[i]) or np.isnan(aucs_transformed[i]))]
    if valid_indices:
        x = np.arange(len(valid_indices))
        width = 0.35
        axes[0, 1].bar(x - width/2, [aucs_original[i] for i in valid_indices], 
                      width, label='Original', alpha=0.7, color='blue')
        axes[0, 1].bar(x + width/2, [aucs_transformed[i] for i in valid_indices], 
                      width, label='Transformed', alpha=0.7, color='orange')
        axes[0, 1].set_title('Disease Classification AUC: Original vs Transformed')
        axes[0, 1].set_xlabel('Disease')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([disease_labels[i] for i in valid_indices], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: Original vs Transformed predictions
    axes[1, 0].scatter(orig_disease_probs.flatten(), trans_disease_probs.flatten(), 
                      alpha=0.5, s=10)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect preservation')
    axes[1, 0].set_title('Disease Predictions: Original vs Transformed')
    axes[1, 0].set_xlabel('Original Predictions')
    axes[1, 0].set_ylabel('Transformed Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dataset-wise analysis
    chexpert_mask = np.array([name == 'CheXpert' for name in dataset_names])
    mimic_mask = np.array([name == 'MIMIC-CXR' for name in dataset_names])
    
    if chexpert_mask.sum() > 0 and mimic_mask.sum() > 0:
        # Calculate average parameters by dataset
        chex_params = transform_params[chexpert_mask].mean(axis=0)
        mimic_params = transform_params[mimic_mask].mean(axis=0)
        
        x = np.arange(4)
        width = 0.35
        axes[1, 1].bar(x - width/2, chex_params, width, label='CheXpert', alpha=0.7)
        axes[1, 1].bar(x + width/2, mimic_params, width, label='MIMIC-CXR', alpha=0.7)
        axes[1, 1].set_title('Average Transformation Parameters by Dataset')
        axes[1, 1].set_xlabel('Parameter')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(param_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    disease_path = os.path.join(output_dir, 'disease_preservation_analysis.png')
    plt.savefig(disease_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {disease_path}")
    
    # 4. SUMMARY STATISTICS
    print("\n📊 SUMMARY STATISTICS:")
    print("="*50)
    
    # Transformation magnitude
    transform_magnitude = np.mean(np.abs(transformed_images - original_images))
    print(f"🔄 Average transformation magnitude: {transform_magnitude:.4f}")
    
    # Parameter statistics
    print(f"\n🎛️ TRANSFORMATION PARAMETERS:")
    for i, param in enumerate(param_names):
        mean_val = np.mean(transform_params[:, i])
        std_val = np.std(transform_params[:, i])
        print(f"   {param}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Disease preservation
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    if valid_corrs:
        avg_correlation = np.mean(valid_corrs)
        print(f"\n🏥 DISEASE PRESERVATION:")
        print(f"   Average correlation: {avg_correlation:.4f}")
        print(f"   Diseases with correlation >0.8: {sum(1 for c in valid_corrs if c > 0.8)}/{len(valid_corrs)}")
    
    # Dataset distribution
    chex_count = chexpert_mask.sum() if 'chexpert_mask' in locals() else 0
    mimic_count = mimic_mask.sum() if 'mimic_mask' in locals() else 0
    print(f"\n📊 DATASET DISTRIBUTION:")
    print(f"   CheXpert samples: {chex_count}")
    print(f"   MIMIC-CXR samples: {mimic_count}")
    
    print(f"\n💾 Visualizations saved to: {output_dir}")
    
    return {
        'transform_magnitude': transform_magnitude,
        'avg_correlation': np.nanmean(correlations),
        'param_stats': {param_names[i]: {'mean': np.mean(transform_params[:, i]), 
                                        'std': np.std(transform_params[:, i])} 
                       for i in range(4)}
    }

if __name__ == "__main__":
    sample_path = "/lotterlab/emily_torchxrayvision/outputs/morphology_autostainer_20251204_153438/samples/samples_epoch_25.npy"
    stats = visualize_autostainer_samples(sample_path)
    print("\n🎉 Visualization complete!")
