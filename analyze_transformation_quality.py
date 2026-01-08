#!/usr/bin/env python3
"""
Deep dive analysis: Is the transformation actually working?
Check if 0.004 magnitude is meaningful or if the model learned to do nothing.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def analyze_transformation_quality():
    """Analyze whether the transformations are meaningful"""
    
    # Load run 1 data (best config)
    sample_file = '/lotterlab/emily_torchxrayvision/outputs/hyperparam_search_20251207_004928/run_001/samples/samples_epoch_15.npy'
    
    print("🔍 Deep Analysis: Is the Transformation Meaningful?")
    print("="*60)
    
    data = np.load(sample_file, allow_pickle=True).item()
    
    original_images = data['original_images']
    transformed_images = data['transformed_images']
    control_points = data['control_points']
    dataset_names = data['dataset_names']
    
    # 1. Check if control points deviate from identity
    print("\n📊 CONTROL POINT ANALYSIS:")
    print("-"*60)
    
    identity_points = np.linspace(0, 1, control_points.shape[1])
    identity_curve = np.stack([identity_points, identity_points], axis=-1)
    
    for dataset_type in ['chexpert', 'mimic']:
        indices = [i for i, name in enumerate(dataset_names) if dataset_type in name.lower()]
        if not indices:
            continue
            
        avg_cp = np.mean(control_points[indices], axis=0)
        deviation = np.mean(np.abs(avg_cp - identity_curve))
        max_deviation = np.max(np.abs(avg_cp - identity_curve))
        
        print(f"\n{dataset_type.upper()}:")
        print(f"  Average deviation from identity: {deviation:.6f}")
        print(f"  Maximum deviation: {max_deviation:.6f}")
        print(f"  Control points Y-values: {avg_cp[:, 1]}")
        
        # Check if it's a meaningful non-linear transformation
        slopes = np.diff(avg_cp[:, 1]) / np.diff(avg_cp[:, 0])
        slope_variance = np.var(slopes)
        print(f"  Slope variance (non-linearity): {slope_variance:.6f}")
        
        if slope_variance < 0.001:
            print(f"  ⚠️  WARNING: Nearly linear! Likely not learning.")
        else:
            print(f"  ✓ Non-linear transformation detected")
    
    # 2. Check histogram changes
    print("\n\n📊 HISTOGRAM ANALYSIS:")
    print("-"*60)
    
    for dataset_type in ['chexpert', 'mimic']:
        indices = [i for i, name in enumerate(dataset_names) if dataset_type in name.lower()][:5]
        if not indices:
            continue
        
        print(f"\n{dataset_type.upper()} (first 5 samples):")
        
        for idx in indices:
            orig = original_images[idx][0]
            trans = transformed_images[idx][0]
            
            # Compute histogram statistics
            orig_mean = np.mean(orig)
            trans_mean = np.mean(trans)
            orig_std = np.std(orig)
            trans_std = np.std(trans)
            
            # KL divergence approximation
            orig_hist, bins = np.histogram(orig.flatten(), bins=50, range=(0, 1), density=True)
            trans_hist, _ = np.histogram(trans.flatten(), bins=bins, density=True)
            
            # Avoid log(0)
            orig_hist = np.clip(orig_hist, 1e-10, None)
            trans_hist = np.clip(trans_hist, 1e-10, None)
            
            kl_div = np.sum(orig_hist * np.log(orig_hist / trans_hist)) * (bins[1] - bins[0])
            
            print(f"  Sample {idx}:")
            print(f"    Mean: {orig_mean:.4f} → {trans_mean:.4f} (Δ={trans_mean-orig_mean:+.4f})")
            print(f"    Std:  {orig_std:.4f} → {trans_std:.4f} (Δ={trans_std-orig_std:+.4f})")
            print(f"    KL Divergence: {kl_div:.6f}")
            
            if abs(trans_mean - orig_mean) < 0.001 and abs(trans_std - orig_std) < 0.001:
                print(f"    ⚠️  Minimal change detected")
            else:
                print(f"    ✓ Meaningful histogram shift")
    
    # 3. Visual check - are there visible differences?
    print("\n\n📊 PIXEL-LEVEL ANALYSIS:")
    print("-"*60)
    
    all_diffs = []
    for i in range(len(original_images)):
        diff = np.abs(transformed_images[i][0] - original_images[i][0])
        all_diffs.append(diff)
        
    all_diffs = np.array(all_diffs)
    
    print(f"Mean absolute difference: {np.mean(all_diffs):.6f}")
    print(f"Median absolute difference: {np.median(all_diffs):.6f}")
    print(f"95th percentile difference: {np.percentile(all_diffs, 95):.6f}")
    print(f"Max difference: {np.max(all_diffs):.6f}")
    
    # Count significantly changed pixels (>0.01 difference)
    significant_changes = (all_diffs > 0.01).sum() / all_diffs.size
    print(f"Percentage of pixels with >0.01 change: {significant_changes*100:.2f}%")
    
    # 4. VERDICT
    print("\n\n" + "="*60)
    print("🎯 VERDICT:")
    print("="*60)
    
    if np.mean(all_diffs) < 0.005:
        print("⚠️  MINIMAL TRANSFORMATION")
        print("   The model is making very subtle changes.")
        print("   This could mean:")
        print("   1. Loss weights are too conservative")
        print("   2. Disease preservation penalty is too strong")
        print("   3. The model found a 'safe' local minimum")
        print("\n   RECOMMENDATION: Try configs with:")
        print("   - LOWER λ_disease (0.5-1.0 instead of 2.0)")
        print("   - LOWER λ_perceptual (0.1-0.5 instead of 1.0)")
        print("   - HIGHER λ_scanner (10-20 instead of 5.0)")
    else:
        print("✓ MEANINGFUL TRANSFORMATION")
        print("   The model is learning useful transformations!")

if __name__ == "__main__":
    analyze_transformation_quality()
