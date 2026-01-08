#!/usr/bin/env python
"""
Hyperparameter Search for AutoStainer Training
Runs multiple configurations of the Fixed Simple AutoStainer
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import argparse

def run_training_config(config, run_id):
    """Run a single training configuration"""

    # Create output directory
    output_dir = f"outputs/hyperparam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Build command
    cmd = [
        "python", "scripts/train_fixed_simple_autostainer.py",
        "--output_dir", output_dir,
        "--num_epochs", str(config.get("num_epochs", 20)),
        "--batch_size", str(config.get("batch_size", 8)),
        "--transformer_lr", str(config.get("transformer_lr", 0.005)),
        "--scanner_lr", str(config.get("scanner_lr", 0.00001)),
        "--disease_lr", str(config.get("disease_lr", 0.0001)),
        "--lambda_adversarial", str(config.get("lambda_adversarial", 50.0)),
        "--lambda_disease", str(config.get("lambda_disease", 2.0)),
        "--lambda_embedding", str(config.get("lambda_embedding", 0.5)),
        "--latent_dim", str(config.get("latent_dim", 128)),
    ]

    print(f"Running config {run_id}: {config}")
    print(f"Command: {' '.join(cmd)}")

    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        success = result.returncode == 0

        # Save output
        with open(f"{output_dir}/stdout.log", 'w') as f:
            f.write(result.stdout)
        with open(f"{output_dir}/stderr.log", 'w') as f:
            f.write(result.stderr)

        return success, output_dir

    except subprocess.TimeoutExpired:
        print(f"Config {run_id} timed out")
        return False, output_dir
    except Exception as e:
        print(f"Config {run_id} failed: {e}")
        return False, output_dir

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for AutoStainer")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of random configurations to try")
    parser.add_argument("--output_base", type=str, default="outputs", help="Base output directory")
    args = parser.parse_args()

    # Define hyperparameter search space
    search_space = {
        "num_epochs": [20, 30, 50],
        "batch_size": [4, 8, 16],
        "transformer_lr": [0.001, 0.005, 0.01],
        "scanner_lr": [0.000001, 0.00001, 0.0001],
        "disease_lr": [0.00001, 0.0001, 0.001],
        "lambda_adversarial": [10.0, 25.0, 50.0, 100.0],
        "lambda_disease": [0.5, 1.0, 2.0, 5.0],
        "lambda_embedding": [0.1, 0.5, 1.0],
        "latent_dim": [64, 128, 256],
    }

    import random
    random.seed(42)  # For reproducibility

    results = []

    for run_id in range(args.num_runs):
        # Sample random configuration
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)

        # Run training
        success, output_dir = run_training_config(config, run_id)

        results.append({
            "run_id": run_id,
            "config": config,
            "success": success,
            "output_dir": output_dir
        })

        print(f"Completed run {run_id + 1}/{args.num_runs}")

    # Save summary
    summary_file = f"{args.output_base}/hyperparam_search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Hyperparameter search complete. Results saved to {summary_file}")

    # Print summary
    successful_runs = sum(1 for r in results if r["success"])
    print(f"Successful runs: {successful_runs}/{len(results)}")

if __name__ == "__main__":
    main()