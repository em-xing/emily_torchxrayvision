"""
Quick diagnostic script to check why disease AUC is 0.0
Run this to understand what's happening during prediction collection
"""

import torch
import numpy as np

# Simulate what happens during training
print("=" * 80)
print("DISEASE AUC DEBUGGING SIMULATION")
print("=" * 80)

# Simulate batch data
batch_size = 8
num_diseases_model = 14  # DenseNet predicts 14 diseases
num_diseases_data = 13   # Dataset might have 13 diseases

print(f"\n1️⃣ STEP 1: Model makes predictions")
print(f"   Batch size: {batch_size}")
print(f"   Model predicts: {num_diseases_model} diseases")
print(f"   Dataset has: {num_diseases_data} disease labels")

# Simulate predictions (logits from model)
disease_logits_orig = torch.randn(batch_size, num_diseases_model)
disease_logits_trans = torch.randn(batch_size, num_diseases_model)
disease_labels = torch.randn(batch_size, num_diseases_data)

print(f"\n   Original predictions shape: {disease_logits_orig.shape}")
print(f"   Transformed predictions shape: {disease_logits_trans.shape}")
print(f"   Labels shape: {disease_labels.shape}")

# Step 2: Align tensors (handle dimension mismatch)
print(f"\n2️⃣ STEP 2: Align dimensions (truncate to match)")

def align_disease_tensors(predictions, labels):
    batch_size = predictions.shape[0]
    pred_diseases = predictions.shape[1]
    label_diseases = labels.shape[1]
    
    if label_diseases < pred_diseases:
        # Truncate predictions to match labels
        aligned_predictions = predictions[:, :label_diseases]
        return aligned_predictions, labels
    else:
        return predictions, labels

aligned_orig, aligned_labels = align_disease_tensors(disease_logits_orig, disease_labels)
aligned_trans, _ = align_disease_tensors(disease_logits_trans, disease_labels)

print(f"   Aligned original shape: {aligned_orig.shape}")
print(f"   Aligned transformed shape: {aligned_trans.shape}")
print(f"   Aligned labels shape: {aligned_labels.shape}")

# Step 3: Convert to probabilities and store
print(f"\n3️⃣ STEP 3: Convert logits to probabilities (sigmoid)")

orig_probs = torch.sigmoid(aligned_orig)
trans_probs = torch.sigmoid(aligned_trans)

print(f"   Original probs shape: {orig_probs.shape}")
print(f"   Transformed probs shape: {trans_probs.shape}")
print(f"   Sample prob value range: [{orig_probs.min():.3f}, {orig_probs.max():.3f}]")

# Convert to numpy
orig_probs_np = orig_probs.cpu().numpy()
trans_probs_np = trans_probs.cpu().numpy()
labels_np = aligned_labels.cpu().numpy()

print(f"\n   NumPy arrays (for sklearn):")
print(f"   orig_probs_np shape: {orig_probs_np.shape} (ndim={orig_probs_np.ndim})")
print(f"   trans_probs_np shape: {trans_probs_np.shape} (ndim={trans_probs_np.ndim})")
print(f"   labels_np shape: {labels_np.shape} (ndim={labels_np.ndim})")

# Check if dimensions are correct
if orig_probs_np.ndim == 2 and labels_np.ndim == 2:
    print(f"   ✅ Dimensions are correct (2D)!")
else:
    print(f"   ❌ ERROR: Wrong dimensions! Expected 2D arrays.")

# Step 4: Simulate collecting multiple batches
print(f"\n4️⃣ STEP 4: Collect from multiple batches")

all_orig_probs = []
all_trans_probs = []
all_labels = []

num_batches = 5
for i in range(num_batches):
    # Simulate batch
    batch_orig = np.random.rand(batch_size, num_diseases_data)
    batch_trans = np.random.rand(batch_size, num_diseases_data)
    batch_labels = np.random.randint(0, 2, size=(batch_size, num_diseases_data)).astype(float)
    
    all_orig_probs.append(batch_orig)
    all_trans_probs.append(batch_trans)
    all_labels.append(batch_labels)
    
    if i < 3:
        print(f"   Batch {i}: probs shape={batch_orig.shape}, labels shape={batch_labels.shape}")

print(f"   Total batches collected: {len(all_orig_probs)}")

# Step 5: Concatenate
print(f"\n5️⃣ STEP 5: Concatenate all batches")

y_pred_orig = np.concatenate(all_orig_probs, axis=0)
y_pred_trans = np.concatenate(all_trans_probs, axis=0)
y_true = np.concatenate(all_labels, axis=0)

print(f"   y_pred_orig shape: {y_pred_orig.shape}")
print(f"   y_pred_trans shape: {y_pred_trans.shape}")
print(f"   y_true shape: {y_true.shape}")

# Step 6: Calculate AUC per disease
print(f"\n6️⃣ STEP 6: Calculate AUC per disease")

from sklearn.metrics import roc_auc_score

auc_scores_orig = []
auc_scores_trans = []

for disease_idx in range(y_true.shape[1]):
    # Check for valid labels (not NaN)
    valid_mask = ~np.isnan(y_true[:, disease_idx])
    
    # Check for at least 2 classes
    n_classes = len(np.unique(y_true[valid_mask, disease_idx]))
    
    if valid_mask.sum() > 0 and n_classes > 1:
        try:
            auc_orig = roc_auc_score(y_true[valid_mask, disease_idx], 
                                     y_pred_orig[valid_mask, disease_idx])
            auc_trans = roc_auc_score(y_true[valid_mask, disease_idx], 
                                      y_pred_trans[valid_mask, disease_idx])
            auc_scores_orig.append(auc_orig)
            auc_scores_trans.append(auc_trans)
            
            if disease_idx < 3:
                print(f"   Disease {disease_idx}: AUC_orig={auc_orig:.3f}, AUC_trans={auc_trans:.3f}")
        except Exception as e:
            print(f"   Disease {disease_idx}: Failed - {e}")
    else:
        print(f"   Disease {disease_idx}: Skipped (valid={valid_mask.sum()}, classes={n_classes})")

# Step 7: Average
print(f"\n7️⃣ STEP 7: Average AUC across diseases")

if auc_scores_orig:
    disease_auc_orig = np.mean(auc_scores_orig)
    disease_auc_trans = np.mean(auc_scores_trans)
    print(f"   ✅ Final AUC (Original): {disease_auc_orig:.3f}")
    print(f"   ✅ Final AUC (Transformed): {disease_auc_trans:.3f}")
    print(f"   📊 Calculated from {len(auc_scores_orig)}/{y_true.shape[1]} diseases")
else:
    print(f"   ❌ ERROR: No valid AUC scores calculated!")
    print(f"   This is why your AUC is 0.0!")

print("\n" + "=" * 80)
print("WHAT TO CHECK IN YOUR ACTUAL CODE:")
print("=" * 80)
print("1. Is 'Number of batches collected' > 0? (Should be ~100s)")
print("2. Are tensor shapes 2D? (e.g., [8, 13], not [104])")
print("3. Do labels have valid values? (not all NaN)")
print("4. Do labels have both 0 and 1? (need positive and negative samples)")
print("5. Is concatenation producing correct shapes?")
print("\nRun your training and check the debug output in epoch 1!")
print("=" * 80)
