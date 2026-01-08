#!/usr/bin/env python3
"""
Visualize samples from the best hyperparameter configuration
Shows original vs transformed images with spline control points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

def plot_spline_curve(control_points, ax):
    """Plot the spline transformation curve"""
    x_coords = control_points[:, 0]
    y_coords = control_points[:, 1]
    
    # Plot identity line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Identity')
    
    # Plot control points
    ax.plot(x_coords, y_coords, 'ro-', linewidth=2, markersize=8, label='Spline Transform')
    
    ax.set_xlabel('Input Intensity', fontsize=10)
    ax.set_ylabel('Output Intensity', fontsize=10)
    ax.set_title('Learned Transformation', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

def visualize_samples(sample_file, output_dir, num_samples=8):
    """Visualize original vs transformed images with control points"""
    
    print(f"📊 Loading samples from: {sample_file}")
    data = np.load(sample_file, allow_pickle=True).item()
    
    original_images = data['original_images']
    transformed_images = data['transformed_images']
    control_points = data['control_points']
    scanner_labels = data['scanner_labels']
    dataset_names = data['dataset_names']
    orig_probs = data['original_disease_probs']
    trans_probs = data['transformed_disease_probs']
    
    print(f"   Total samples: {len(original_images)}")
    print(f"   Image shape: {original_images[0].shape}")
    print(f"   Datasets: {set(dataset_names)}")
    
    # CRITICAL: Check and normalize image ranges
    print(f"   Original images range: [{original_images.min():.3f}, {original_images.max():.3f}]")
    print(f"   Transformed images range: [{transformed_images.min():.3f}, {transformed_images.max():.3f}]")
    
    # Normalize images to [0, 1] if needed
    if original_images.max() > 1.0 or original_images.min() < 0.0:
        print("   ⚠️  Normalizing original images to [0, 1]")
        orig_min, orig_max = original_images.min(), original_images.max()
        if orig_max > orig_min:
            original_images = (original_images - orig_min) / (orig_max - orig_min)
    
    if transformed_images.max() > 1.0 or transformed_images.min() < 0.0:
        print("   ⚠️  Normalizing transformed images to [0, 1]")
        trans_min, trans_max = transformed_images.min(), transformed_images.max()
        if trans_max > trans_min:
            transformed_images = (transformed_images - trans_min) / (trans_max - trans_min)
    
    # Select samples to visualize (mix of both datasets)
    indices = []
    chexpert_idx = [i for i, name in enumerate(dataset_names) if 'chexpert' in name.lower()]
    mimic_idx = [i for i, name in enumerate(dataset_names) if 'mimic' in name.lower()]
    
    # Get 4 from each dataset
    indices.extend(chexpert_idx[:4] if len(chexpert_idx) >= 4 else chexpert_idx)
    indices.extend(mimic_idx[:4] if len(mimic_idx) >= 4 else mimic_idx)
    indices = indices[:num_samples]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * len(indices)))
    gs = gridspec.GridSpec(len(indices), 4, figure=fig, hspace=0.3, wspace=0.3)
    
    disease_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                     'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                     'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    for plot_idx, sample_idx in enumerate(indices):
        orig_img = original_images[sample_idx][0]  # [1, H, W] -> [H, W]
        trans_img = transformed_images[sample_idx][0]
        cp = control_points[sample_idx]
        dataset = dataset_names[sample_idx]
        scanner_label = scanner_labels[sample_idx]
        
        # Ensure images are in [0, 1] range for this sample
        if orig_img.max() > 1.0:
            orig_img = orig_img / orig_img.max()
        if trans_img.max() > 1.0:
            trans_img = trans_img / trans_img.max()
        
        # Calculate image difference (now both are properly normalized)
        diff_img = np.abs(trans_img - orig_img)
        
        # Original image
        ax1 = fig.add_subplot(gs[plot_idx, 0])
        im1 = ax1.imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Original\n{dataset} (Scanner {scanner_label})', 
                     fontsize=10, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Add histogram
        ax1_hist = ax1.inset_axes([0.05, 0.05, 0.3, 0.2])
        ax1_hist.hist(orig_img.flatten(), bins=50, alpha=0.7, color='blue', range=(0, 1))
        ax1_hist.set_xticks([])
        ax1_hist.set_yticks([])
        ax1_hist.patch.set_alpha(0.5)
        
        # Transformed image
        ax2 = fig.add_subplot(gs[plot_idx, 1])
        im2 = ax2.imshow(trans_img, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Transformed\nΔ = {np.mean(diff_img):.4f}', 
                     fontsize=10, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Add histogram
        ax2_hist = ax2.inset_axes([0.05, 0.05, 0.3, 0.2])
        ax2_hist.hist(trans_img.flatten(), bins=50, alpha=0.7, color='green', range=(0, 1))
        ax2_hist.set_xticks([])
        ax2_hist.set_yticks([])
        ax2_hist.patch.set_alpha(0.5)
        
        # Difference image
        ax3 = fig.add_subplot(gs[plot_idx, 2])
        im3 = ax3.imshow(diff_img, cmap='hot', vmin=0, vmax=0.2)
        ax3.set_title(f'Absolute Difference\nMax: {np.max(diff_img):.4f}', 
                     fontsize=10, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Spline curve
        ax4 = fig.add_subplot(gs[plot_idx, 3])
        plot_spline_curve(cp, ax4)
        
        # Add disease prediction comparison
        orig_top3 = np.argsort(orig_probs[sample_idx])[-3:][::-1]
        trans_top3 = np.argsort(trans_probs[sample_idx])[-3:][::-1]
        
        disease_text = "Top 3 Diseases:\n"
        for i, (o_idx, t_idx) in enumerate(zip(orig_top3, trans_top3)):
            if o_idx < len(disease_labels):
                disease_text += f"Orig: {disease_labels[o_idx][:15]}: {orig_probs[sample_idx][o_idx]:.2f}\n"
                disease_text += f"Trans: {disease_labels[t_idx][:15]}: {trans_probs[sample_idx][t_idx]:.2f}\n"
                if i < 2:
                    disease_text += "---\n"
        
        ax4.text(0.5, -0.35, disease_text, transform=ax4.transAxes,
                fontsize=7, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Best Configuration Samples (Config #1)\n' + 
                f'Scanner Acc: 54.0% | Disease Pres: 98.2% | Transform Mag: 0.0040',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'best_config_samples_{os.path.basename(sample_file)[:-4]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization: {output_file}")
    plt.close()

def main():
    """Main visualization function"""
    
    sample_dir = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/run_000/samples'
    output_dir = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/visualizations'
    
    print("🎨 Visualizing Best Configuration Samples")
    print(f"📁 Sample directory: {sample_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find all sample files
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.npy')])
    
    if not sample_files:
        print("❌ No sample files found!")
        return
    
    print(f"📊 Found {len(sample_files)} sample files")
    
    # Visualize each sample file
    for sample_file in sample_files:
        full_path = os.path.join(sample_dir, sample_file)
        visualize_samples(full_path, output_dir, num_samples=8)
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
