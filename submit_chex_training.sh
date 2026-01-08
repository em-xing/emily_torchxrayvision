#!/bin/bash
#SBATCH --job-name=chex_windowing_training
#SBATCH --output=/lotterlab/emily_torchxrayvision/outputs/chex_windowing_viz/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Activate your virtual environment
source .venv/bin/activate

# Change to the project directory
cd /lotterlab/emily_torchxrayvision

# Run the training script with windowing and enhanced checkpointing
python3 scripts/train_model.py \
    --dataset chex \
    --name chex_windowing_viz \
    --window_nbins 64 \
    --visualize_spline \
    --gpu 0 \
    --num_epochs 50 \
    --batch_size 16 \
    --fixed_splits \
    --use_scheduler
