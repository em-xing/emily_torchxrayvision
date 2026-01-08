# Disease Prediction AUC Metric

## Overview
Added **Disease Prediction AUC (Area Under ROC Curve)** metric to measure actual disease classification accuracy, replacing the less meaningful correlation-based "disease preservation" metric.

## What Changed

### Old Metric: Disease Preservation (Correlation)
```
Disease Preservation: 92.8%
```
- **What it measured**: Correlation between original and transformed predictions
- **Problem**: High correlation doesn't mean accurate predictions!
  - Example: If model predicts `[0.1, 0.2, 0.3]` for original and `[0.11, 0.21, 0.31]` for transformed, correlation = 100% even if both are wrong!
  - Just means predictions "move together", not that they're correct

### New Metric: Disease Prediction AUC
```
Disease Prediction AUC:
   Original Images: 85.3%
   Transformed Images: 84.7%
```
- **What it measures**: Actual disease classification accuracy (AUC-ROC)
- **Why it's better**: 
  - Compares predictions to **ground truth labels**
  - Standard metric in medical imaging (80%+ AUC is good, 90%+ is excellent)
  - Shows if transformations hurt prediction accuracy

## How It Works

1. **Collect predictions throughout epoch**:
   - Original image predictions: `y_pred_orig`
   - Transformed image predictions: `y_pred_trans`
   - Ground truth labels: `y_true`

2. **Calculate AUC for each disease**:
   ```python
   for each disease:
       auc_orig = roc_auc_score(y_true[:, disease], y_pred_orig[:, disease])
       auc_trans = roc_auc_score(y_true[:, disease], y_pred_trans[:, disease])
   ```

3. **Average across all diseases** (macro-average):
   ```python
   disease_auc_original = mean(auc_scores_orig)
   disease_auc_transformed = mean(auc_scores_trans)
   ```

## Interpreting Results

### AUC Score Ranges
- **90-100%**: Excellent (pretrained model should be here)
- **80-90%**: Good
- **70-80%**: Fair
- **60-70%**: Poor
- **50-60%**: Barely better than random
- **<50%**: Worse than random

### Expected Results

**Original Images (Pretrained DenseNet121)**:
```
Disease Prediction AUC:
   Original Images: 80-90%  ← Pretrained model should be high
```

**After Transformation**:
```
Disease Prediction AUC:
   Transformed Images: 78-88%  ← Should stay close to original
```

**Good transformation**:
- Transform AUC ≥ 75% (maintains clinical utility)
- Drop < 5% from original (minimal accuracy loss)
- Scanner accuracy → 50-55% (good confusion)

**Bad transformation**:
- Transform AUC < 70% (loses disease information)
- Drop > 10% from original (too destructive)

## Comparison with Old Metric

### Example Scenario 1: Tiny Transformations
```
Old Metric:
   Disease Preservation: 98%  ← Looks great!
   Transform Magnitude: 0.01  ← But changes nothing!
   
New Metric:
   Original AUC: 85%
   Transformed AUC: 85%  ← Same as original (no surprise)
   Scanner Acc: 95%  ← Not confusing scanner
```
**Interpretation**: High correlation just means "didn't change much"

### Example Scenario 2: Strong Transformations
```
Old Metric:
   Disease Preservation: 75%  ← Looks okay
   Transform Magnitude: 0.15  ← Strong changes
   
New Metric:
   Original AUC: 85%
   Transformed AUC: 82%  ← Only 3% drop (good!)
   Scanner Acc: 52%  ← Successfully confusing scanner
```
**Interpretation**: AUC shows transformations preserve clinical accuracy

### Example Scenario 3: Destructive Transformations
```
Old Metric:
   Disease Preservation: 65%  ← Looks bad
   Transform Magnitude: 0.25  ← Very strong
   
New Metric:
   Original AUC: 85%
   Transformed AUC: 60%  ← 25% drop (too much!)
   Scanner Acc: 48%  ← Good confusion but destroyed features
```
**Interpretation**: AUC shows transformations are too aggressive

## Implementation Details

### Data Collection
```python
# During training
all_orig_disease_probs = []
all_trans_disease_probs = []
all_disease_labels = []

for batch in dataloader:
    # Get predictions
    orig_probs = model(original_images)
    trans_probs = model(transformed_images)
    
    # Collect for AUC calculation
    all_orig_disease_probs.append(orig_probs)
    all_trans_disease_probs.append(trans_probs)
    all_disease_labels.append(labels)
```

### AUC Calculation
```python
# At end of epoch
y_pred_orig = np.concatenate(all_orig_disease_probs)
y_pred_trans = np.concatenate(all_trans_disease_probs)
y_true = np.concatenate(all_disease_labels)

# Calculate per-disease AUC
for disease_idx in range(num_diseases):
    valid_mask = ~np.isnan(y_true[:, disease_idx])
    auc_orig = roc_auc_score(y_true[valid_mask], y_pred_orig[valid_mask])
    auc_trans = roc_auc_score(y_true[valid_mask], y_pred_trans[valid_mask])

# Average across diseases
disease_auc_original = mean(auc_scores_orig)
disease_auc_transformed = mean(auc_scores_trans)
```

## Benefits

1. **✅ Meaningful**: Measures actual prediction accuracy against ground truth
2. **✅ Standard**: AUC-ROC is the gold standard in medical imaging
3. **✅ Comparable**: Can compare to published results (DenseNet121-All reports ~80-90% AUC)
4. **✅ Interpretable**: Clear threshold for "good enough" (>75% for clinical use)
5. **✅ Actionable**: Can optimize hyperparameters to maximize transformed AUC

## Hyperparameter Search Integration

The hyperparameter search can now optimize for:
```python
combined_score = (
    0.4 * scanner_confusion_score +     # Scanner acc → 50%
    0.4 * transformed_auc_score +       # Transformed AUC > 75%
    0.2 * auc_drop_penalty              # Minimize AUC drop
)
```

This ensures we find configurations that:
- Confuse the scanner (scanner acc → 50%)
- Preserve clinical accuracy (transformed AUC > 75%)
- Minimize information loss (AUC drop < 5%)

## Current Status
- ✅ AUC metric implemented in training script
- ✅ Displayed in epoch summary
- ✅ Saved in metrics JSON
- 🔄 Currently running in hyperparameter search
- ⏳ Will need to update hyperparameter search scoring to use AUC instead of correlation

## Next Steps
1. Monitor AUC in current hyperparameter search
2. Update scoring function to prioritize high transformed AUC
3. Set threshold: configs with transformed AUC < 70% should be deprioritized
4. Analyze best configs: scanner confusion + high AUC + low AUC drop
