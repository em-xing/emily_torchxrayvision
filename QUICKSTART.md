# Quick Start Guide

## Installation & Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/emily_torchxrayvision.git
cd emily_torchxrayvision

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Core Workflows

### 1. Find Best Hyperparameters 
```bash
python scripts/hyperparameter_search.py
```
- Tests 216+ configurations automatically
- Includes early stopping for efficiency
- Results saved to `outputs/hyperparam_search_*/`

### 2. Train with Specific Configuration
```bash
python scripts/train_fixed_simple_autostainer.py
```
- Modify config dict in script for custom parameters
- Trains for 15 epochs (or until early stopping)
- Outputs: models, samples, metrics

### 3. Analyze Training Results, Generate Reports, Validate
```bash
# Analyze disease classification accuracy
python analyze_disease_accuracy.py --input_dir outputs/hyperparam_search_20240111_120000

# Analyze hyperparameter search results across all runs
python analyze_hyperparam_results.py --results_dir outputs/ --top_k 10

# Assess transformation quality metrics
python analyze_transformation_quality.py --model_path outputs/best_run/models/final_model.pth

# Create summary of best configurations
python create_best_config_summary.py --search_results outputs/hyperparam_search_*/results.json

# Visualize samples from best configuration
python visualize_best_config_samples.py --config_path outputs/best_config.json --num_samples 20

# Compare loss curves between different runs
python visualize_loss_comparison.py --run_dirs outputs/run1,outputs/run2,outputs/run3

# Test AUC loss implementation
python test_auc_loss.py --batch_size 16 --num_epochs 5

# Validate pretrained model accuracy on test set
python test_pretrained_accuracy.py --model_path models/pretrained_chexpert.pth --dataset mimic

# Debug AUC calculation issues
python debug_auc_calculation.py --predictions_file outputs/predictions.npy --targets_file outputs/targets.npy
```

### 4. Advanced Training Options
```bash
# Train with custom configuration file
python scripts/train_fixed_simple_autostainer.py --config config/custom_config.json

# Resume training from checkpoint
python scripts/train_fixed_simple_autostainer.py --resume outputs/checkpoint_epoch_10.pth

# Train with different batch size for memory constraints
python scripts/train_fixed_simple_autostainer.py --batch_size 2

# Run hyperparameter search with reduced configurations for testing
python scripts/hyperparameter_search.py --num_configs 10 --max_epochs 5
```

### 5. Dataset and Model Utilities
```bash
# Inspect dataset samples
python scripts/inspect_samples.py --dataset chexpert --num_samples 5

# Process and prepare images
python scripts/process_image.py --input_image path/to/xray.jpg --output_dir processed/

# Calibrate model predictions
python scripts/model_calibrate.py --model_path models/trained_model.pth --calibration_method temperature_scaling

# Extract features from images
python scripts/xray_representations.ipynb  # Run in Jupyter for feature extraction
```

## Key Configuration Parameters
- `transformer_lr: 0.005` (aggressive learning for domain adaptation)
- `scanner_lr: 0.00001` (conservative for stable confusion)
- `lambda_adversarial: 50.0` (strong scanner confusion priority)
- `early_stop_scanner_acc: 0.85` (stop if scanner acc stays >85% for 3 epochs)

## Expected Outputs
- `outputs/*/models/`: Trained model checkpoints
- `outputs/*/samples/`: Transformed X-ray images
- `outputs/*/metrics.json`: Training metrics and AUC scores
- `outputs/*/config.json`: Configuration used

## Quick Troubleshooting
- **CUDA issues**: Check `torch.cuda.is_available()`
- **Memory errors**: Reduce batch size to 4 or 2
- **Dataset errors**: Verify CheXpert/MIMIC-CXR paths
- **No convergence**: Adjust learning rates or loss weights

## Typical Workflow
1. Run hyperparameter search to find promising configs
2. Train best config with `train_fixed_simple_autostainer.py`
3. Analyze results with visualization scripts
4. Generate summary reports

