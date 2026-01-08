"""
Comprehensive analysis of hyperparameter search results.
Analyzes multiple hyperparameter search runs and generates detailed reports.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def find_hyperparam_searches(base_dir: str = "outputs") -> List[Path]:
    """Find all hyperparam search directories."""
    base_path = Path(base_dir)
    search_dirs = []
    
    for path in sorted(base_path.glob("hyperparam_search_*")):
        if path.is_dir():
            # Check if it has results
            if (path / "search_results.csv").exists():
                search_dirs.append(path)
    
    return search_dirs


def load_search_results(search_dir: Path) -> pd.DataFrame:
    """Load search results from a directory."""
    csv_path = search_dir / "search_results.csv"
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    df['search_dir'] = search_dir.name
    df['timestamp'] = search_dir.name.split('_')[-2] + '_' + search_dir.name.split('_')[-1]
    
    return df


def load_detailed_metrics(search_dir: Path, run_id: int) -> Dict:
    """Load detailed metrics for a specific run."""
    metrics_path = search_dir / f"run_{run_id:03d}" / "analysis" / "metrics.json"
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def analyze_all_searches(output_dir: str = "hyperparam_analysis"):
    """Analyze all hyperparameter search results."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all searches
    search_dirs = find_hyperparam_searches()
    
    if not search_dirs:
        print("No hyperparameter search results found!")
        return
    
    print(f"Found {len(search_dirs)} hyperparameter search runs")
    print("=" * 80)
    
    # Load all results
    all_results = []
    for search_dir in search_dirs:
        df = load_search_results(search_dir)
        if df is not None and len(df) > 0:
            # Filter out incomplete runs (all zeros)
            valid_df = df[df['combined_score'] > 0]
            if len(valid_df) > 0:
                all_results.append(valid_df)
                print(f"\n{search_dir.name}:")
                print(f"  Total runs: {len(df)}")
                print(f"  Valid runs: {len(valid_df)}")
    
    if not all_results:
        print("\nNo valid results found!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n{'=' * 80}")
    print(f"Total valid runs across all searches: {len(combined_df)}")
    
    # Generate comprehensive report
    generate_summary_report(combined_df, output_dir)
    
    # Create visualizations
    create_visualizations(combined_df, output_dir)
    
    # Analyze top configurations
    analyze_top_configs(combined_df, search_dirs, output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"Analysis complete! Results saved to: {output_dir}/")


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive summary report."""
    
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER SEARCH SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total runs analyzed: {len(df)}\n")
        f.write(f"Number of search sessions: {df['search_dir'].nunique()}\n\n")
        
        # Metric ranges
        f.write("METRIC RANGES\n")
        f.write("-" * 80 + "\n")
        metrics = ['combined_score', 'scanner_accuracy', 'realfake_loss', 
                   'hypernet_loss', 'transform_magnitude']
        
        for metric in metrics:
            if metric in df.columns:
                f.write(f"\n{metric}:\n")
                f.write(f"  Min:    {df[metric].min():.6f}\n")
                f.write(f"  Max:    {df[metric].max():.6f}\n")
                f.write(f"  Mean:   {df[metric].mean():.6f}\n")
                f.write(f"  Median: {df[metric].median():.6f}\n")
                f.write(f"  Std:    {df[metric].std():.6f}\n")
        
        # Disease preservation or AUC
        if 'disease_preservation' in df.columns:
            f.write(f"\ndisease_preservation:\n")
            f.write(f"  Min:    {df['disease_preservation'].min():.6f}\n")
            f.write(f"  Max:    {df['disease_preservation'].max():.6f}\n")
            f.write(f"  Mean:   {df['disease_preservation'].mean():.6f}\n")
            f.write(f"  Median: {df['disease_preservation'].median():.6f}\n")
        
        if 'disease_auc_original' in df.columns and 'disease_auc_transformed' in df.columns:
            f.write(f"\ndisease_auc_original:\n")
            f.write(f"  Min:    {df['disease_auc_original'].min():.6f}\n")
            f.write(f"  Max:    {df['disease_auc_original'].max():.6f}\n")
            f.write(f"  Mean:   {df['disease_auc_original'].mean():.6f}\n")
            
            f.write(f"\ndisease_auc_transformed:\n")
            f.write(f"  Min:    {df['disease_auc_transformed'].min():.6f}\n")
            f.write(f"  Max:    {df['disease_auc_transformed'].max():.6f}\n")
            f.write(f"  Mean:   {df['disease_auc_transformed'].mean():.6f}\n")
        
        # Top 10 configurations
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 10 CONFIGURATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        top_10 = df.nlargest(10, 'combined_score')
        
        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"#{idx} - Combined Score: {row['combined_score']:.6f}\n")
            f.write(f"     Search: {row['search_dir']}\n")
            f.write(f"     Run ID: {row['run_id']}\n")
            f.write(f"     Scanner Accuracy: {row['scanner_accuracy']:.6f}\n")
            
            if 'disease_preservation' in row:
                f.write(f"     Disease Preservation: {row['disease_preservation']:.6f}\n")
            if 'disease_auc_original' in row and 'disease_auc_transformed' in row:
                f.write(f"     Disease AUC (Original): {row['disease_auc_original']:.6f}\n")
                f.write(f"     Disease AUC (Transformed): {row['disease_auc_transformed']:.6f}\n")
            
            f.write(f"     RealFake Loss: {row['realfake_loss']:.6f}\n")
            f.write(f"     Hypernet Loss: {row['hypernet_loss']:.6f}\n")
            f.write(f"     Transform Magnitude: {row['transform_magnitude']:.6f}\n")
            f.write(f"     Hyperparameters:\n")
            f.write(f"       hypernet_lr: {row['hypernet_lr']}\n")
            f.write(f"       scanner_lr: {row['scanner_lr']}\n")
            f.write(f"       lambda_realfake: {row['lambda_realfake']}\n")
            f.write(f"       lambda_scanner: {row['lambda_scanner']}\n")
            f.write(f"       lambda_perceptual: {row['lambda_perceptual']}\n")
            f.write(f"       lambda_disease: {row['lambda_disease']}\n")
            f.write(f"       lambda_smooth: {row['lambda_smooth']}\n")
            f.write("\n")
        
        # Hyperparameter impact analysis
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER IMPACT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        hyperparam_cols = ['hypernet_lr', 'scanner_lr', 'lambda_realfake', 
                          'lambda_scanner', 'lambda_perceptual', 'lambda_disease', 
                          'lambda_smooth']
        
        for param in hyperparam_cols:
            if param in df.columns:
                f.write(f"\n{param}:\n")
                grouped = df.groupby(param)['combined_score'].agg(['mean', 'std', 'count'])
                f.write(grouped.to_string())
                f.write("\n")
        
    print(f"\nSummary report saved to: {report_path}")


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations."""
    
    print("\nGenerating visualizations...")
    
    # 1. Combined score distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['combined_score'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Combined Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Combined Scores Across All Runs')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Scanner accuracy vs disease preservation/AUC
    plt.figure(figsize=(12, 6))
    
    if 'disease_preservation' in df.columns:
        scatter = plt.scatter(df['scanner_accuracy'], df['disease_preservation'], 
                            c=df['combined_score'], cmap='viridis', s=100, alpha=0.6)
        plt.xlabel('Scanner Accuracy')
        plt.ylabel('Disease Preservation')
        plt.title('Scanner Accuracy vs Disease Preservation')
    elif 'disease_auc_transformed' in df.columns:
        scatter = plt.scatter(df['scanner_accuracy'], df['disease_auc_transformed'], 
                            c=df['combined_score'], cmap='viridis', s=100, alpha=0.6)
        plt.xlabel('Scanner Accuracy')
        plt.ylabel('Disease AUC (Transformed)')
        plt.title('Scanner Accuracy vs Disease AUC (Transformed)')
    
    plt.colorbar(scatter, label='Combined Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_disease.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Hyperparameter impact on combined score
    hyperparam_cols = ['lambda_realfake', 'lambda_scanner', 'lambda_disease']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, param in enumerate(hyperparam_cols):
        if param in df.columns:
            param_scores = df.groupby(param)['combined_score'].mean().sort_index()
            axes[idx].plot(param_scores.index, param_scores.values, 'o-', linewidth=2, markersize=8)
            axes[idx].set_xlabel(param)
            axes[idx].set_ylabel('Mean Combined Score')
            axes[idx].set_title(f'Impact of {param}')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparam_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Transform magnitude vs performance
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(df['transform_magnitude'], df['combined_score'], 
                         c=df['scanner_accuracy'], cmap='coolwarm', s=100, alpha=0.6)
    plt.xlabel('Transform Magnitude')
    plt.ylabel('Combined Score')
    plt.title('Transform Magnitude vs Performance')
    plt.colorbar(scatter, label='Scanner Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig(os.path.join(output_dir, 'transform_magnitude.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Loss correlation heatmap
    loss_cols = ['realfake_loss', 'hypernet_loss', 'scanner_accuracy', 'combined_score']
    if 'disease_preservation' in df.columns:
        loss_cols.append('disease_preservation')
    elif 'disease_auc_transformed' in df.columns:
        loss_cols.extend(['disease_auc_original', 'disease_auc_transformed'])
    
    available_cols = [col for col in loss_cols if col in df.columns]
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[available_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Learning rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'hypernet_lr' in df.columns:
        lr_scores = df.groupby('hypernet_lr')['combined_score'].agg(['mean', 'std'])
        axes[0].errorbar(lr_scores.index, lr_scores['mean'], yerr=lr_scores['std'], 
                        fmt='o-', linewidth=2, markersize=8, capsize=5)
        axes[0].set_xlabel('Hypernet Learning Rate')
        axes[0].set_ylabel('Combined Score')
        axes[0].set_title('Hypernet Learning Rate Impact')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
    
    if 'scanner_lr' in df.columns:
        lr_scores = df.groupby('scanner_lr')['combined_score'].agg(['mean', 'std'])
        axes[1].errorbar(lr_scores.index, lr_scores['mean'], yerr=lr_scores['std'], 
                        fmt='o-', linewidth=2, markersize=8, capsize=5)
        axes[1].set_xlabel('Scanner Learning Rate')
        axes[1].set_ylabel('Combined Score')
        axes[1].set_title('Scanner Learning Rate Impact')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}/")


def analyze_top_configs(df: pd.DataFrame, search_dirs: List[Path], output_dir: str):
    """Detailed analysis of top configurations with training curves."""
    
    print("\nAnalyzing top configurations...")
    
    top_5 = df.nlargest(5, 'combined_score')
    
    # Load detailed metrics for top 5
    detailed_metrics = []
    for _, row in top_5.iterrows():
        search_dir = [d for d in search_dirs if d.name == row['search_dir']][0]
        metrics = load_detailed_metrics(search_dir, int(row['run_id']))
        if metrics:
            detailed_metrics.append({
                'config': row,
                'metrics': metrics
            })
    
    if not detailed_metrics:
        print("Could not load detailed metrics for top configurations")
        return
    
    # Plot training curves for top 5
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metric_names = ['hypernet_loss', 'realfake_loss', 'scanner_accuracy', 
                   'disease_preservation', 'transform_magnitude', 'combined_score']
    
    # Handle different metric names
    if detailed_metrics[0]['metrics'].get('disease_preservation') is None:
        metric_names[3] = 'disease_auc_transformed'
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        for i, item in enumerate(detailed_metrics):
            metrics = item['metrics']
            config = item['config']
            
            # Handle different metric names
            if metric_name in metrics:
                epochs = metrics.get('epochs', list(range(1, len(metrics[metric_name]) + 1)))
                values = metrics[metric_name]
                
                label = f"#{i+1} (λ_rf={config['lambda_realfake']}, λ_sc={config['lambda_scanner']})"
                ax.plot(epochs, values, 'o-', label=label, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Over Training')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top5_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison table
    comparison_path = os.path.join(output_dir, 'top5_comparison.csv')
    top_5.to_csv(comparison_path, index=False)
    print(f"Top 5 comparison saved to: {comparison_path}")
    
    # Generate recommendations
    recommendations_path = os.path.join(output_dir, 'recommendations.txt')
    with open(recommendations_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS BASED ON HYPERPARAMETER SEARCH\n")
        f.write("=" * 80 + "\n\n")
        
        best_config = top_5.iloc[0]
        
        f.write("BEST CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Combined Score: {best_config['combined_score']:.6f}\n")
        f.write(f"Scanner Accuracy: {best_config['scanner_accuracy']:.6f}\n")
        
        if 'disease_preservation' in best_config:
            f.write(f"Disease Preservation: {best_config['disease_preservation']:.6f}\n")
        if 'disease_auc_transformed' in best_config:
            f.write(f"Disease AUC (Transformed): {best_config['disease_auc_transformed']:.6f}\n")
        
        f.write(f"\nRecommended Hyperparameters:\n")
        f.write(f"  hypernet_lr: {best_config['hypernet_lr']}\n")
        f.write(f"  scanner_lr: {best_config['scanner_lr']}\n")
        f.write(f"  lambda_realfake: {best_config['lambda_realfake']}\n")
        f.write(f"  lambda_scanner: {best_config['lambda_scanner']}\n")
        f.write(f"  lambda_perceptual: {best_config['lambda_perceptual']}\n")
        f.write(f"  lambda_disease: {best_config['lambda_disease']}\n")
        f.write(f"  lambda_smooth: {best_config['lambda_smooth']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("=" * 80 + "\n\n")
        
        # Analyze patterns
        # Learning rates
        avg_hypernet_lr_top5 = top_5['hypernet_lr'].mean()
        avg_scanner_lr_top5 = top_5['scanner_lr'].mean()
        f.write(f"1. Learning Rates:\n")
        f.write(f"   - Top configurations use hypernet_lr around {avg_hypernet_lr_top5}\n")
        f.write(f"   - Top configurations use scanner_lr around {avg_scanner_lr_top5}\n\n")
        
        # Lambda values
        f.write(f"2. Loss Weights:\n")
        for param in ['lambda_realfake', 'lambda_scanner', 'lambda_disease']:
            if param in top_5.columns:
                values = top_5[param].unique()
                f.write(f"   - {param}: {values}\n")
        f.write("\n")
        
        # Trade-offs
        f.write(f"3. Performance Trade-offs:\n")
        f.write(f"   - Scanner accuracy range in top 5: {top_5['scanner_accuracy'].min():.4f} - {top_5['scanner_accuracy'].max():.4f}\n")
        
        if 'disease_preservation' in top_5.columns:
            f.write(f"   - Disease preservation range: {top_5['disease_preservation'].min():.4f} - {top_5['disease_preservation'].max():.4f}\n")
        if 'disease_auc_transformed' in top_5.columns:
            f.write(f"   - Disease AUC range: {top_5['disease_auc_transformed'].min():.4f} - {top_5['disease_auc_transformed'].max():.4f}\n")
        
        f.write(f"   - Transform magnitude range: {top_5['transform_magnitude'].min():.6f} - {top_5['transform_magnitude'].max():.6f}\n")
        
    print(f"Recommendations saved to: {recommendations_path}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("HYPERPARAMETER SEARCH ANALYSIS")
    print("=" * 80)
    
    analyze_all_searches()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - hyperparam_analysis/summary_report.txt")
    print("  - hyperparam_analysis/recommendations.txt")
    print("  - hyperparam_analysis/top5_comparison.csv")
    print("  - hyperparam_analysis/*.png (visualizations)")


if __name__ == "__main__":
    main()
