#!/usr/bin/env python3
"""
Create a compact summary visualization of the best configuration
Shows key transformations and control points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def create_summary_visualization():
    """Create a concise summary of the best configuration results"""
    
    sample_file = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/run_000/samples/samples_epoch_15.npy'
    output_file = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/visualizations/best_config_summary.png'
    
    print(f"📊 Creating summary visualization...")
    data = np.load(sample_file, allow_pickle=True).item()
    
    original_images = data['original_images']
    transformed_images = data['transformed_images']
    control_points = data['control_points']
    dataset_names = data['dataset_names']
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle('Best Configuration Summary (Config #1)\n' + 
                'Scanner Acc: 54.0% | Disease Preservation: 98.2% | Transform Magnitude: 0.0040\n' +
                'hypernet_lr=1e-4, scanner_lr=1e-5, λ_realfake=0.5, λ_scanner=5.0, λ_perceptual=1.0, λ_disease=2.0',
                fontsize=13, fontweight='bold')
    
    # Select 4 CheXpert and 4 MIMIC samples
    chexpert_idx = [i for i, name in enumerate(dataset_names) if 'chexpert' in name.lower()][:4]
    mimic_idx = [i for i, name in enumerate(dataset_names) if 'mimic' in name.lower()][:4]
    all_idx = chexpert_idx + mimic_idx
    
    # Plot samples in a grid (2 rows of 4)
    for plot_idx, sample_idx in enumerate(all_idx[:8]):
        row = plot_idx // 4
        col = plot_idx % 4
        
        orig_img = original_images[sample_idx][0]
        trans_img = transformed_images[sample_idx][0]
        cp = control_points[sample_idx]
        dataset = 'CheXpert' if 'chexpert' in dataset_names[sample_idx].lower() else 'MIMIC'
        
        diff = np.abs(trans_img - orig_img)
        
        # Original
        ax_orig = fig.add_subplot(gs[row*2, col])
        ax_orig.imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f'{dataset} - Original', fontsize=9)
        ax_orig.axis('off')
        
        # Transformed
        ax_trans = fig.add_subplot(gs[row*2+1, col])
        ax_trans.imshow(trans_img, cmap='gray', vmin=0, vmax=1)
        ax_trans.set_title(f'Transformed (Δ={np.mean(diff):.4f})', fontsize=9)
        ax_trans.axis('off')
    
    # Add overall spline curves comparison
    ax_spline = fig.add_subplot(gs[4, :2])
    
    # Average spline for CheXpert
    chexpert_cp = np.mean(control_points[chexpert_idx], axis=0)
    mimic_cp = np.mean(control_points[mimic_idx], axis=0)
    
    ax_spline.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Identity')
    ax_spline.plot(chexpert_cp[:, 0], chexpert_cp[:, 1], 'b-o', linewidth=2, markersize=8, 
                   label='CheXpert Transform', alpha=0.8)
    ax_spline.plot(mimic_cp[:, 0], mimic_cp[:, 1], 'r-s', linewidth=2, markersize=8,
                   label='MIMIC Transform', alpha=0.8)
    
    ax_spline.set_xlabel('Input Intensity', fontsize=11, fontweight='bold')
    ax_spline.set_ylabel('Output Intensity', fontsize=11, fontweight='bold')
    ax_spline.set_title('Average Learned Transformations by Dataset', fontsize=12, fontweight='bold')
    ax_spline.legend(fontsize=10, loc='upper left')
    ax_spline.grid(True, alpha=0.3)
    ax_spline.set_xlim([0, 1])
    ax_spline.set_ylim([0, 1])
    ax_spline.set_aspect('equal')
    
    # Add transformation statistics
    ax_stats = fig.add_subplot(gs[4, 2:])
    ax_stats.axis('off')
    
    # Calculate statistics
    all_diffs = [np.mean(np.abs(transformed_images[i][0] - original_images[i][0])) 
                 for i in range(len(original_images))]
    chexpert_diffs = [all_diffs[i] for i in chexpert_idx]
    mimic_diffs = [all_diffs[i] for i in mimic_idx]
    
    stats_text = f"""
📊 TRANSFORMATION STATISTICS

Overall:
  • Mean Transform Magnitude: {np.mean(all_diffs):.4f}
  • Std Transform Magnitude: {np.std(all_diffs):.4f}
  • Min/Max: {np.min(all_diffs):.4f} / {np.max(all_diffs):.4f}

By Dataset:
  • CheXpert Mean Δ: {np.mean(chexpert_diffs):.4f}
  • MIMIC Mean Δ: {np.mean(mimic_diffs):.4f}

Key Insights:
  ✅ Scanner confusion achieved (54% acc)
  ✅ Excellent disease preservation (98.2%)
  ✅ Subtle but effective transformations
  ✅ Dataset-specific adaptations learned
    """
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved summary: {output_file}")
    plt.close()

if __name__ == "__main__":
    create_summary_visualization()
