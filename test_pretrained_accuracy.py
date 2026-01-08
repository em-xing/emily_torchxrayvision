#!/usr/bin/env python3
"""
Test the accuracy of the pretrained disease classifier
on a sample of real CheXpert and MIMIC data
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import os

sys.path.insert(0, '.')
from scripts.train_fixed_simple_autostainer import DiseaseClassifier
from scripts.gan_datasets import MultiDatasetGANLoader

def test_pretrained_accuracy():
    """Test pretrained classifier on real data"""
    
    print("🧪 Testing Pretrained Disease Classifier Accuracy\n")
    print("=" * 80)
    
    # Load disease classifier
    classifier = DiseaseClassifier(input_channels=1, num_diseases=18)
    classifier = classifier.to('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.eval()
    
    device = next(classifier.parameters()).device
    print(f"📍 Device: {device}\n")
    
    # Load a small sample of real data
    print("📊 Loading test data...")
    loader = MultiDatasetGANLoader(limit_per_dataset=500, batch_size=16)
    train_dataloader, val_dataloader = loader.create_dataloaders(train_split=0.8, num_workers=2)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    all_datasets = []
    
    print("\n🔬 Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images = batch['img'].to(device)
            labels = batch['lab'].to(device)
            dataset_names = batch['dataset_name']
            
            # Normalize images to [0, 1] if needed
            img_min, img_max = images.min(), images.max()
            if img_max > 1.0 or img_min < 0.0:
                images = (images - img_min) / (img_max - img_min + 1e-8)
            
            # Get predictions
            predictions, _ = classifier(images, return_features=False)
            predictions = torch.sigmoid(predictions)  # Convert to probabilities
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_datasets.extend(dataset_names)
            
            if batch_idx >= 10:  # Limit to ~160 samples for quick test
                break
    
    # Concatenate results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\n✅ Collected {len(all_predictions)} samples")
    print(f"   Predictions shape: {all_predictions.shape}")
    print(f"   Labels shape: {all_labels.shape}")
    
    # Calculate AUC for each disease
    print("\n" + "=" * 80)
    print("📊 Disease-Specific AUC (Area Under ROC Curve)")
    print("=" * 80)
    print(f"{'Disease':<35} {'Samples':<10} {'AUC':>10}")
    print("-" * 80)
    
    disease_names = classifier.model.pathologies
    aucs = []
    
    for disease_idx, disease_name in enumerate(disease_names):
        if disease_idx >= all_labels.shape[1]:
            continue
            
        # Get valid samples (non-NaN labels)
        valid_mask = ~np.isnan(all_labels[:, disease_idx])
        
        if valid_mask.sum() < 10:
            print(f"{disease_name:<35} {valid_mask.sum():<10} Insufficient data")
            continue
        
        # Check if we have both classes
        labels_valid = all_labels[valid_mask, disease_idx]
        if len(np.unique(labels_valid)) < 2:
            print(f"{disease_name:<35} {valid_mask.sum():<10} Only one class")
            continue
        
        preds_valid = all_predictions[valid_mask, disease_idx]
        
        try:
            auc = roc_auc_score(labels_valid, preds_valid)
            aucs.append(auc)
            marker = "🎯" if auc > 0.8 else "✅" if auc > 0.7 else "⚠️"
            print(f"{disease_name:<35} {valid_mask.sum():<10} {auc:>10.4f}  {marker}")
        except:
            print(f"{disease_name:<35} {valid_mask.sum():<10} Error calculating")
    
    print("=" * 80)
    
    if len(aucs) > 0:
        mean_auc = np.mean(aucs)
        print(f"\n📈 Average AUC: {mean_auc:.4f}")
        
        if mean_auc > 0.8:
            print("🎯 EXCELLENT: Pretrained classifier has high accuracy!")
        elif mean_auc > 0.7:
            print("✅ GOOD: Pretrained classifier has decent accuracy")
        else:
            print("⚠️ MODERATE: Pretrained classifier has modest accuracy")
    else:
        print("\n⚠️ Could not calculate AUC scores")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_pretrained_accuracy()
