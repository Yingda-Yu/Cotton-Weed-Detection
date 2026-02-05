#!/usr/bin/env python3
"""
Compare training metrics between baseline and cleaned models
Generate visualization charts showing changes in various metrics

Usage:
    python tools/compare_training_metrics.py \
        --baseline runs/detect/yolov8n_baseline_fast27/results.csv \
        --cleaned runs/detect/yolov8n_cleaned_fast/results.csv \
        --output comparison_metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Metric groups
METRIC_GROUPS = {
    "loss": {
        "train/box_loss": "Train Box Loss",
        "train/cls_loss": "Train Classification Loss",
        "train/dfl_loss": "Train DFL Loss",
        "val/box_loss": "Val Box Loss",
        "val/cls_loss": "Val Classification Loss",
        "val/dfl_loss": "Val DFL Loss"
    },
    "metrics": {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@0.5",
        "metrics/mAP50-95(B)": "mAP@0.5:0.95"
    },
    "learning_rate": {
        "lr/pg0": "Learning Rate (pg0)",
        "lr/pg1": "Learning Rate (pg1)",
        "lr/pg2": "Learning Rate (pg2)"
    }
}

# Colors for baseline vs cleaned
COLORS = {
    "baseline": "#1f77b4",
    "cleaned": "#2ca02c"
}


def load_results(csv_file):
    """Load training results CSV file"""
    df = pd.read_csv(csv_file)
    return df


def plot_loss_comparison(baseline_df, cleaned_df, output_dir):
    """Plot loss comparison charts"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training and Validation Loss Comparison', fontsize=16, fontweight='bold')
    
    loss_metrics = [
        ("train/box_loss", "Train Box Loss", axes[0, 0]),
        ("train/cls_loss", "Train Classification Loss", axes[0, 1]),
        ("train/dfl_loss", "Train DFL Loss", axes[0, 2]),
        ("val/box_loss", "Val Box Loss", axes[1, 0]),
        ("val/cls_loss", "Val Classification Loss", axes[1, 1]),
        ("val/dfl_loss", "Val DFL Loss", axes[1, 2])
    ]
    
    for metric, title, ax in loss_metrics:
        if metric in baseline_df.columns and metric in cleaned_df.columns:
            baseline_epochs = range(1, len(baseline_df) + 1)
            cleaned_epochs = range(1, len(cleaned_df) + 1)
            
            ax.plot(baseline_epochs, baseline_df[metric], 
                   label='Baseline', 
                   color=COLORS["baseline"], 
                   linewidth=2, 
                   marker='o', 
                   markersize=4)
            ax.plot(cleaned_epochs, cleaned_df[metric], 
                   label='Cleaned', 
                   color=COLORS["cleaned"], 
                   linewidth=2, 
                   marker='s', 
                   markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "loss_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loss comparison chart saved: {output_file}")
    plt.close()


def plot_metrics_comparison(baseline_df, cleaned_df, output_dir):
    """Plot evaluation metrics comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ("metrics/precision(B)", "Precision", axes[0, 0]),
        ("metrics/recall(B)", "Recall", axes[0, 1]),
        ("metrics/mAP50(B)", "mAP@0.5", axes[1, 0]),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95", axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        if metric in baseline_df.columns and metric in cleaned_df.columns:
            baseline_epochs = range(1, len(baseline_df) + 1)
            cleaned_epochs = range(1, len(cleaned_df) + 1)
            
            ax.plot(baseline_epochs, baseline_df[metric], 
                   label='Baseline', 
                   color=COLORS["baseline"], 
                   linewidth=2.5, 
                   marker='o', 
                   markersize=5)
            ax.plot(cleaned_epochs, cleaned_df[metric], 
                   label='Cleaned', 
                   color=COLORS["cleaned"], 
                   linewidth=2.5, 
                   marker='s', 
                   markersize=5)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_file = output_dir / "metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison chart saved: {output_file}")
    plt.close()


def plot_individual_metric(baseline_df, cleaned_df, metric, title, output_dir):
    """Plot individual metric comparison chart"""
    if metric not in baseline_df.columns or metric not in cleaned_df.columns:
        print(f"Metric {metric} not found, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_epochs = range(1, len(baseline_df) + 1)
    cleaned_epochs = range(1, len(cleaned_df) + 1)
    
    ax.plot(baseline_epochs, baseline_df[metric], 
           label='Baseline', 
           color=COLORS["baseline"], 
           linewidth=2.5, 
           marker='o', 
           markersize=6,
           alpha=0.8)
    ax.plot(cleaned_epochs, cleaned_df[metric], 
           label='Cleaned', 
           color=COLORS["cleaned"], 
           linewidth=2.5, 
           marker='s', 
           markersize=6,
           alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Value', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis range for metrics
    if 'metrics' in metric:
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # Generate safe filename
    safe_metric_name = metric.replace('/', '_').replace('(', '').replace(')', '')
    output_file = output_dir / f"{safe_metric_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{title} chart saved: {output_file}")
    plt.close()


def plot_final_comparison(baseline_df, cleaned_df, output_dir):
    """Plot final performance comparison bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Final Performance Comparison', fontsize=16, fontweight='bold')
    
    # Get final values
    baseline_final = baseline_df.iloc[-1]
    cleaned_final = cleaned_df.iloc[-1]
    
    # mAP comparison
    ax = axes[0]
    metrics_map = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    labels = ['mAP@0.5', 'mAP@0.5:0.95']
    baseline_values = [baseline_final[m] if m in baseline_final else 0 for m in metrics_map]
    cleaned_values = [cleaned_final[m] if m in cleaned_final else 0 for m in metrics_map]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', 
                   color=COLORS["baseline"], alpha=0.8)
    bars2 = ax.bar(x + width/2, cleaned_values, width, label='Cleaned', 
                   color=COLORS["cleaned"], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('mAP Metrics Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(baseline_values), max(cleaned_values)) * 1.2])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Precision and Recall comparison
    ax = axes[1]
    metrics_pr = ['metrics/precision(B)', 'metrics/recall(B)']
    labels_pr = ['Precision', 'Recall']
    baseline_values_pr = [baseline_final[m] if m in baseline_final else 0 for m in metrics_pr]
    cleaned_values_pr = [cleaned_final[m] if m in cleaned_final else 0 for m in metrics_pr]
    
    x = np.arange(len(labels_pr))
    
    bars1 = ax.bar(x - width/2, baseline_values_pr, width, label='Baseline', 
                   color=COLORS["baseline"], alpha=0.8)
    bars2 = ax.bar(x + width/2, cleaned_values_pr, width, label='Cleaned', 
                   color=COLORS["cleaned"], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision & Recall Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_pr)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(baseline_values_pr), max(cleaned_values_pr)) * 1.2])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "final_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Final performance comparison chart saved: {output_file}")
    plt.close()


def generate_comparison_report(baseline_df, cleaned_df, output_dir):
    """Generate comparison report"""
    baseline_final = baseline_df.iloc[-1]
    cleaned_final = cleaned_df.iloc[-1]
    
    report = {
        "baseline": {
            "epochs": len(baseline_df),
            "final_metrics": {}
        },
        "cleaned": {
            "epochs": len(cleaned_df),
            "final_metrics": {}
        },
        "improvement": {}
    }
    
    # Extract final metrics
    for metric in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                   'metrics/precision(B)', 'metrics/recall(B)']:
        if metric in baseline_final and metric in cleaned_final:
            baseline_val = float(baseline_final[metric])
            cleaned_val = float(cleaned_final[metric])
            improvement = cleaned_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            
            report["baseline"]["final_metrics"][metric] = baseline_val
            report["cleaned"]["final_metrics"][metric] = cleaned_val
            report["improvement"][metric] = {
                "absolute": improvement,
                "percentage": improvement_pct
            }
    
    # Save report
    report_file = output_dir / "comparison_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison report saved: {report_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Performance Comparison Summary")
    print("=" * 70)
    print(f"\nBaseline Model (trained {len(baseline_df)} epochs):")
    if 'metrics/mAP50(B)' in baseline_final:
        print(f"  mAP@0.5: {baseline_final['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in baseline_final:
        print(f"  mAP@0.5:0.95: {baseline_final['metrics/mAP50-95(B)']:.4f}")
    
    print(f"\nCleaned Model (trained {len(cleaned_df)} epochs):")
    if 'metrics/mAP50(B)' in cleaned_final:
        print(f"  mAP@0.5: {cleaned_final['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in cleaned_final:
        print(f"  mAP@0.5:0.95: {cleaned_final['metrics/mAP50-95(B)']:.4f}")
    
    print(f"\nPerformance Improvement:")
    if 'metrics/mAP50(B)' in report["improvement"]:
        imp = report["improvement"]['metrics/mAP50(B)']
        print(f"  mAP@0.5: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)")
    if 'metrics/mAP50-95(B)' in report["improvement"]:
        imp = report["improvement"]['metrics/mAP50-95(B)']
        print(f"  mAP@0.5:0.95: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)")
    print("=" * 70)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare training metrics between baseline and cleaned models and generate visualization charts"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="runs/detect/yolov8n_baseline_fast27/results.csv",
        help="Path to baseline model results.csv"
    )
    parser.add_argument(
        "--cleaned",
        type=str,
        default="runs/detect/yolov8n_cleaned_fast/results.csv",
        help="Path to cleaned model results.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_metrics",
        help="Output directory (default: comparison_metrics)"
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Generate individual charts for each metric"
    )
    
    args = parser.parse_args()
    
    # Check files
    baseline_path = Path(args.baseline)
    cleaned_path = Path(args.cleaned)
    
    if not baseline_path.exists():
        print(f"Baseline results file not found: {baseline_path}")
        return 1

    if not cleaned_path.exists():
        print(f"Cleaned results file not found: {cleaned_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Training Metrics Comparison Tool")
    print("=" * 70)
    print(f"Baseline results: {baseline_path}")
    print(f"Cleaned results: {cleaned_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Load data
    print("\nLoading training results...")
    baseline_df = load_results(baseline_path)
    cleaned_df = load_results(cleaned_path)
    
    print(f"Baseline: {len(baseline_df)} epochs")
    print(f"Cleaned: {len(cleaned_df)} epochs")
    
    # Generate charts
    print("\nGenerating comparison charts...")
    
    plot_loss_comparison(baseline_df, cleaned_df, output_dir)
    plot_metrics_comparison(baseline_df, cleaned_df, output_dir)
    plot_final_comparison(baseline_df, cleaned_df, output_dir)

    if args.individual:
        print("\nGenerating individual metric charts...")
        for metric, title in METRIC_GROUPS["loss"].items():
            plot_individual_metric(baseline_df, cleaned_df, metric, title, output_dir)
        for metric, title in METRIC_GROUPS["metrics"].items():
            plot_individual_metric(baseline_df, cleaned_df, metric, title, output_dir)

    generate_comparison_report(baseline_df, cleaned_df, output_dir)
    
    print("\n" + "=" * 70)
    print("All charts generated successfully.")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nGenerated charts:")
    print("  - loss_comparison.png (Loss comparison)")
    print("  - metrics_comparison.png (Metrics comparison)")
    print("  - final_comparison.png (Final performance comparison)")
    if args.individual:
        print("  - Individual charts for each metric")
    print("  - comparison_report.json (Comparison report)")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
