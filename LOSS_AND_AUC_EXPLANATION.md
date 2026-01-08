# How Disease AUC and Losses are Calculated

## Overview
This document explains how the disease AUC metric and various losses are calculated in the AutoStainer training pipeline.

---

## 🎯 Disease AUC Calculation

### What is Disease AUC?
**AUC (Area Under the ROC Curve)** measures the ability of the disease classifier to distinguish between positive and negative cases for each disease. It's a value between 0 and 1:
- **1.0** = Perfect classification
- **0.5** = Random guessing
- **0.0** = Perfectly wrong

### How It's Calculated (Lines 560-695)

#### Step 1: Collect Predictions During Training
For each batch, we collect:
```python
# Line 567-586: Collect predictions per batch
if aligned_orig.numel() > 0:
    orig_probs_np = torch.sigmoid(aligned_orig).cpu().numpy()  # [batch, num_diseases]
    trans_probs_np = torch.sigmoid(aligned_trans).cpu().numpy() # [batch, num_diseases]
    labels_np = aligned_disease_labels.cpu().numpy()           # [batch, num_diseases]
    
    # Store as lists of 2D arrays
    all_orig_disease_probs.append(orig_probs_np)
    all_trans_disease_probs.append(trans_probs_np)
    all_disease_labels.append(labels_np)
```

**Key Points:**
- `aligned_orig`: Disease predictions for **original** images (logits)
- `aligned_trans`: Disease predictions for **transformed** images (logits)
- `torch.sigmoid()`: Converts logits to probabilities [0, 1]
- Each array is shape `[batch_size, num_diseases]` (e.g., [8, 13] or [8, 14])

#### Step 2: Concatenate All Batches (End of Epoch)
```python
# Line 642-644: Combine all batches
y_pred_orig = np.concatenate(all_orig_disease_probs, axis=0)  # [N_samples, num_diseases]
y_pred_trans = np.concatenate(all_trans_disease_probs, axis=0)
y_true = np.concatenate(all_disease_labels, axis=0)
```

Result: Large arrays with all samples from the epoch
- Example: [10000, 13] if we had 10,000 samples with 13 diseases

#### Step 3: Calculate Per-Disease AUC
```python
# Line 665-680: Calculate AUC for each disease independently
for disease_idx in range(y_true.shape[1]):
    # Get valid labels (not NaN)
    valid_mask = ~np.isnan(y_true[:, disease_idx])
    
    # Need at least 2 classes (positive and negative) to compute AUC
    if valid_mask.sum() > 0 and len(np.unique(y_true[valid_mask, disease_idx])) > 1:
        auc_orig = roc_auc_score(y_true[valid_mask, disease_idx], 
                                 y_pred_orig[valid_mask, disease_idx])
        auc_trans = roc_auc_score(y_true[valid_mask, disease_idx], 
                                  y_pred_trans[valid_mask, disease_idx])
        auc_scores_orig.append(auc_orig)
        auc_scores_trans.append(auc_trans)
```

**Why per-disease?**
- Each disease is independent (Pneumonia, Cardiomegaly, etc.)
- Some samples have NaN labels (disease not labeled in that dataset)
- We calculate AUC only for samples where that disease is labeled

#### Step 4: Average Across Diseases (Macro-Average)
```python
# Line 690-691: Final AUC score
disease_auc_orig = np.mean(auc_scores_orig)
disease_auc_trans = np.mean(auc_scores_trans)
```

**Example:**
```
Disease 0 (Atelectasis):     AUC = 0.85
Disease 1 (Cardiomegaly):    AUC = 0.90
Disease 2 (Consolidation):   AUC = 0.80
...
Average AUC = 0.85
```

---

## 📊 Loss Functions

The model uses **multiple loss functions** to balance different objectives. Here's how each is calculated:

### 1. Real/Fake Discriminator Loss (Lines 411-426)

**Purpose:** Train a discriminator to distinguish original vs transformed images (ensures transformations look realistic).

```python
# Real images should be classified as "real" (1)
real_preds = self.realfake_discriminator(images)
real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))

# Transformed images should be classified as "fake" (0)
fake_preds = self.realfake_discriminator(transformed_images)
fake_loss = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))

realfake_loss = (real_loss + fake_loss) / 2
```

**Loss Type:** Binary Cross-Entropy (BCE)
- Trains discriminator to output 1 for real, 0 for fake
- Lower loss = better discrimination

---

### 2. Scanner Classifier Loss (Lines 445-461)

**Purpose:** Train scanner classifier to identify which dataset/scanner an image came from.

```python
# Train on both original and transformed images
scanner_logits_real = self.scanner_classifier(images)
scanner_loss_real = self.scanner_criterion(scanner_logits_real, scanner_labels)

scanner_logits_fake = self.scanner_classifier(transformed_images)
scanner_loss_fake = self.scanner_criterion(scanner_logits_fake, scanner_labels)

scanner_loss = scanner_loss_real + scanner_loss_fake
```

**Loss Type:** Cross-Entropy Loss
- Classifies images into N scanner types (e.g., 2 scanners → 2 classes)
- Lower loss = better scanner identification

**Important:** We want the scanner to be **confused** on transformed images (accuracy ~50%), but this loss trains it to be accurate. The hypernetwork will later try to fool it!

---

### 3. Hypernetwork Losses (Lines 474-530)

The hypernetwork (spline transformation) is trained with **5 combined losses**:

#### 3a. Adversarial Real/Fake Loss (Lines 484-486)
```python
# Try to FOOL the discriminator (make transformed images look real)
fake_preds = self.realfake_discriminator(transformed_images)
adversarial_realfake_loss = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
```
- **Goal:** Make discriminator think transformed images are real
- Lower loss = transformations look more realistic

#### 3b. Adversarial Scanner Loss (Lines 488-490)
```python
# Try to FOOL the scanner classifier (confuse it about dataset origin)
scanner_logits_gen = self.scanner_classifier(transformed_images)
adversarial_scanner_loss = self.scanner_criterion(scanner_logits_gen, scanner_labels)
```
- **Goal:** Make scanner classifier confused (we want ~50% accuracy)
- This is the **opposite** of training the scanner in step 2!

#### 3c. Perceptual Loss (Multi-Scale) (Lines 492-501)
```python
# Extract features from multiple layers of the disease classifier
disease_logits_trans, trans_features = self.disease_classifier(
    transformed_images, return_features=True
)

# Compare features at multiple scales (early, mid, late layers)
perceptual_loss = 0
for orig_feat, trans_feat in zip(orig_features, trans_features):
    perceptual_loss += self.l1_loss(orig_feat.detach(), trans_feat)
perceptual_loss = perceptual_loss / len(orig_features)
```
- **Loss Type:** L1 (Mean Absolute Error) on deep features
- **Goal:** Keep internal representations similar (preserve semantic content)
- Compares features from DenseNet layers (not just final predictions)

#### 3d. Disease AUC Loss (Differentiable AUC Optimization) (Lines 497-504)
```python
# Directly optimize for AUC using pairwise ranking loss
aligned_orig, aligned_labels = self.align_disease_tensors(disease_logits.detach(), disease_labels)
aligned_trans, _ = self.align_disease_tensors(disease_logits_trans, disease_labels)

if disease_mask.any() and aligned_orig.numel() > 0:
    disease_auc_loss = self.compute_differentiable_auc_loss(
        aligned_trans[disease_mask], 
        aligned_labels[disease_mask]
    )
```
- **Loss Type:** Pairwise Ranking Loss (differentiable AUC approximation)
- **Goal:** Maximize AUC directly - transformed images should rank positive cases higher than negative cases
- **How it works:**
  - For each disease, compare all (positive, negative) sample pairs
  - Penalize when negative samples are ranked higher than positive samples
  - Uses smooth softplus: `loss = log(1 + exp(-(pos_prob - neg_prob)))`
  - Directly optimizes the ranking quality that AUC measures
- Only computed for valid labels (not NaN)
- **Better than L1 consistency:** Directly optimizes what we measure (AUC), not just similarity

#### 3e. Spline Smoothness Regularization (Line 512-513)
```python
# Penalize large jumps between adjacent control points
spline_smoothness = torch.mean(torch.abs(control_points[:, 1:, :] - control_points[:, :-1, :]))
```
- **Loss Type:** L1 on consecutive control point differences
- **Goal:** Encourage smooth transformations (avoid sharp discontinuities)

#### Combined Hypernetwork Loss (Lines 527-532)
```python
hypernet_loss = (
    lambda_realfake * adversarial_realfake_loss +      # Weight: 1.0
    lambda_scanner * adversarial_scanner_loss +        # Weight: 5.0
    lambda_perceptual * perceptual_loss +              # Weight: 1.0
    lambda_disease * disease_auc_loss +                # Weight: 2.0 (now optimizes AUC directly!)
    lambda_smooth * spline_smoothness                  # Weight: 0.01
)
```

**Current Weights (from config):**
- `lambda_realfake`: 1.0 (realistic appearance)
- `lambda_scanner`: 5.0 (confuse scanner - highest weight!)
- `lambda_perceptual`: 1.0 (preserve features)
- `lambda_disease`: 2.0 (preserve disease predictions)
- `lambda_smooth`: 0.01 (smoothness regularization)

---

## 🔍 Why Disease AUC Might Be 0.0

Based on the code analysis, disease AUC could be 0.0 if:

### Issue 1: No Valid Predictions Collected
```python
# Line 567: Collection condition
if aligned_orig.numel() > 0:
    # Store predictions...
```
If `aligned_orig` is empty for ALL batches, nothing gets stored → AUC = 0.0

**Possible causes:**
- `align_disease_tensors()` returns empty tensors
- All disease labels are NaN

### Issue 2: Wrong Tensor Dimensions
```python
# Line 574-586: Dimension check
if orig_probs_np.ndim != 2 or trans_probs_np.ndim != 2 or labels_np.ndim != 2:
    # Skip this batch
```
If tensors are flattened (1D) instead of 2D, they're skipped → AUC = 0.0

### Issue 3: Concatenation Fails
```python
# Line 642: If this fails or produces wrong shape
y_pred_orig = np.concatenate(all_orig_disease_probs, axis=0)
```
If concatenation produces 1D arrays or fails, AUC calculation is skipped

### Issue 4: No Valid Disease Labels
```python
# Line 668: Need at least 2 classes to compute AUC
if valid_mask.sum() > 0 and len(np.unique(y_true[valid_mask, disease_idx])) > 1:
```
If all labels are NaN or all same value (e.g., all 0s), AUC can't be computed

---

## 🐛 Debugging Steps

To find why AUC is 0.0, check the debug output (epoch 1):

```
🔍 AUC Debug - Before concatenation:
   Number of batches collected: X  <-- Should be > 0
   Batch 0: probs shape=(8, 13), labels shape=(8, 13)  <-- Should be 2D

🔍 AUC Debug - After concatenation:
   y_pred_orig shape: (N, 13)  <-- Should be 2D with N samples
   y_pred_trans shape: (N, 13)
   y_true shape: (N, 13)
```

If you see warnings like:
- "Number of batches collected: 0" → Problem in collection (line 567-586)
- "probs shape=(104,)" → Tensors are flattened (dimension problem)
- "Invalid tensor dimensions" → Check `align_disease_tensors()`
- "Skipping batch X due to dimension mismatch" → All batches skipped!

---

## 📝 Summary

**Disease AUC:**
- Measures actual prediction accuracy (not just preservation)
- Calculated per-disease, then averaged
- Requires valid 2D tensors: `[num_samples, num_diseases]`

**Losses:**
1. **Real/Fake Loss:** Is transformation realistic?
2. **Scanner Loss:** Can we identify the scanner?
3. **Adversarial Real/Fake:** Fool discriminator (make fake look real)
4. **Adversarial Scanner:** Fool scanner (confuse it)
5. **Perceptual Loss:** Preserve deep features
6. **Disease AUC Loss:** Directly optimize for disease prediction accuracy (ranking quality)
7. **Smoothness:** Keep splines smooth

The model balances these competing objectives to achieve:
- 🎯 Scanner confusion (~50% accuracy)
- 🎯 Realistic transformations (fool discriminator)
- 🎯 High disease AUC (preserve medical accuracy)
