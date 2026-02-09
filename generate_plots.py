"""
Generate visualization plots for Qwen SQL fine-tuning evaluation results.

This script creates professional 2D and 3D visualizations:
1. 2D bar chart comparing parseable rate and exact match across models
2. 3D bar chart showing multi-metric comparison (NEW)
3. 3D scatter plot showing accuracy-latency-size trade-off (NEW)
4. 2D scatter plot showing accuracy-latency trade-off
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Evaluation results data
results_data = {
    'model': [
        'qwen_baseline_fp16',
        'qwen_finetuned_fp16',
        'qwen_finetuned_int8',
        'qwen_finetuned_fp16_strict',
        'qwen_finetuned_int8_strict',
        'gpt-4o-mini',
        'claude-3.5-haiku'
    ],
    'parseable_rate': [1.00, 0.93, 0.93, 1.00, 0.99, 1.00, 0.99],
    'exact_match_rate': [0.09, 0.13, 0.13, 0.15, 0.20, 0.04, 0.07],
    'lat_mean': [0.405, 0.527, 2.672, 0.433, 2.152, 1.616, 1.735],
    'lat_p50': [0.422, 0.711, 3.454, 0.427, 2.541, 1.551, 1.541],
    'lat_p95': [0.624, 0.739, 3.623, 0.736, 3.610, 2.820, 2.697],
    'model_size_gb': [6.0, 6.0, 3.0, 6.0, 3.0, 0.0, 0.0]  # 0 for API models
}

df = pd.DataFrame(results_data)

# Material design color palette with gradient
material_colors = {
    'baseline': '#78909C',      # Blue Grey 400
    'finetuned_fp16': '#42A5F5', # Blue 400
    'finetuned_int8': '#AB47BC', # Purple 400
    'api': '#EF5350'             # Red 400
}

# Performance gradient (blue to red)
def get_performance_color(value: float, vmin: float = 0.0, vmax: float = 1.0):
    """Get color based on performance gradient (blue=good, red=bad)."""
    norm_value = (value - vmin) / (vmax - vmin)
    # Invert: high values = blue (good), low values = red (bad)
    norm_value = 1 - norm_value
    cmap = LinearSegmentedColormap.from_list('performance', ['#EF5350', '#FFA726', '#66BB6A', '#42A5F5'])
    return cmap(norm_value)

def assign_color(model_name):
    """Assign material design color based on model type."""
    if 'baseline' in model_name:
        return material_colors['baseline']
    elif 'int8' in model_name:
        return material_colors['finetuned_int8']
    elif 'finetuned' in model_name:
        return material_colors['finetuned_fp16']
    else:
        return material_colors['api']

df['color'] = df['model'].apply(assign_color)
df['display_name'] = df['model'].str.replace('qwen_', '').str.replace('_', ' ').str.title()


def plot_accuracy_comparison():
    """Plot 1: 2D Parseable rate and exact match comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Parseable Rate
    bars1 = ax1.barh(df['display_name'], df['parseable_rate'], color=df['color'], alpha=0.85, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Parseable SQL Rate', fontsize=13, fontweight='bold')
    ax1.set_title('SQL Parseability Comparison\n(Primary Metric)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0.9, 1.01)
    ax1.axvline(x=0.95, color='#D32F2F', linestyle='--', alpha=0.4, linewidth=2, label='95% threshold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, df['parseable_rate'])):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontweight='bold', fontsize=10)
    
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#FAFAFA')
    
    # Exact Match Rate
    bars2 = ax2.barh(df['display_name'], df['exact_match_rate'], color=df['color'], alpha=0.85, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Exact Match Rate', fontsize=13, fontweight='bold')
    ax2.set_title('Exact Match Accuracy\n(Secondary Metric)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 0.25)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, df['exact_match_rate'])):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontweight='bold', fontsize=10)
    
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig('images/results_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: images/results_comparison.png")
    plt.close()


def plot_3d_bar_comparison():
    """Plot 2: 3D bar chart showing multi-metric comparison (NEW)."""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Data preparation
    num_models = len(df)
    x_pos = np.arange(num_models)
    
    # Three metrics to show
    metrics = ['parseable_rate', 'exact_match_rate', 'normalized_latency']
    
    # Normalize latency (inverse so higher = better, scale to 0-1)
    max_lat = df['lat_mean'].max()
    df['normalized_latency'] = 1 - (df['lat_mean'] / (max_lat * 1.5))  # Inverse and scale
    df['normalized_latency'] = df['normalized_latency'].clip(0, 1)
    
    width = 0.25
    depth = 0.5
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        y_pos = np.ones(num_models) * i
        z_pos = np.zeros(num_models)
        
        # Get performance gradient colors for this metric
        colors = [get_performance_color(val) for val in df[metric]]
        
        ax.bar3d(x_pos, y_pos, z_pos, width, depth, df[metric], 
                color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Customize axes
    ax.set_xlabel('Model', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Score', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('3D Multi-Metric Model Comparison\n(Higher bars = Better performance)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set ticks
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels(df['display_name'], rotation=15, ha='right', fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Parseable\nRate', 'Exact\nMatch', 'Speed\n(inverse latency)'], fontsize=10)
    ax.set_zlim(0, 1.1)
    
    # Optimal viewing angle
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    plt.savefig('images/model_comparison_3d.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: images/model_comparison_3d.png")
    plt.close()


def plot_3d_scatter():
    """Plot 3: 3D scatter plot showing accuracy-latency-size trade-off (NEW)."""
    fig = plt.figure(figsize=(14, 10))
    ax: Any = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    x = df['lat_mean']
    y = df['exact_match_rate']
    z = df['parseable_rate']
    
    # Size based on model size (larger bubble = larger model)
    # API models get medium size
    sizes = []
    for size_gb in df['model_size_gb']:
        if size_gb == 0:  # API
            sizes.append(400)
        elif size_gb == 3:  # INT8
            sizes.append(600)
        else:  # FP16
            sizes.append(800)
    
    # Colors: material design with gradient based on exact match
    colors = [get_performance_color(val, vmin=0, vmax=0.25) for val in df['exact_match_rate']]
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, s=sizes, c=colors, alpha=0.8, 
                        edgecolors='black', linewidth=2, depthshade=True)
    
    # Add labels for each point
    for idx, row in df.iterrows():
        ax.text(row['lat_mean'], row['exact_match_rate'], row['parseable_rate'],
               f"  {row['display_name']}", fontsize=8, ha='left', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Labels and title
    ax.set_xlabel('Mean Latency (seconds)', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_ylabel('Exact Match Rate', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_zlabel('Parseable SQL Rate', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_title('3D Performance Trade-off Analysis\n(Lower-right-top corner is optimal)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Optimal viewing angle for this data
    ax.view_init(elev=20, azim=135)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#F5F5F5')
    
    # Add legend for bubble sizes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
              markersize=np.sqrt(800/20), label='FP16 (6GB)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
              markersize=np.sqrt(600/20), label='INT8 (3GB)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
              markersize=np.sqrt(400/20), label='API (Cloud)', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('images/accuracy_latency_3d.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: images/accuracy_latency_3d.png")
    plt.close()


def plot_accuracy_vs_latency():
    """Plot 4: 2D Accuracy-latency trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with material colors
    for idx, row in df.iterrows():
        ax.scatter(row['lat_mean'], row['exact_match_rate'], 
                  s=600, c=row['color'], alpha=0.8, edgecolors='black', linewidth=2, zorder=3)
        
        # Add model labels with smart positioning
        offset_x = 0.05 if 'int8' not in row['model'] else 0.12
        offset_y = 0.005
        ax.annotate(row['display_name'], 
                   (row['lat_mean'], row['exact_match_rate']),
                   xytext=(offset_x, offset_y), 
                   textcoords='offset points',
                   fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1))
    
    ax.set_xlabel('Mean Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exact Match Rate', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Latency Trade-off\n(Lower-right is optimal: high accuracy, low latency)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add reference lines
    ax.axhline(y=0.15, color='#66BB6A', linestyle='--', alpha=0.5, linewidth=2, label='Target accuracy (15%)')
    ax.axvline(x=1.0, color='#FFA726', linestyle='--', alpha=0.5, linewidth=2, label='1s latency threshold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig('images/accuracy_vs_latency.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: images/accuracy_vs_latency.png")
    plt.close()


def generate_summary_table():
    """Generate a formatted summary table."""
    summary = df[['display_name', 'parseable_rate', 'exact_match_rate', 'lat_mean']].copy()
    summary.columns = ['Model', 'Parseable SQL', 'Exact Match', 'Mean Latency (s)']
    
    # Highlight best values
    best_parseable = summary['Parseable SQL'].max()
    best_exact = summary['Exact Match'].max()
    best_latency = summary['Mean Latency (s)'].min()
    
    print("\n" + "="*85)
    print("EVALUATION SUMMARY")
    print("="*85)
    print(summary.to_string(index=False))
    print("="*85)
    print(f"\nBest Parseable SQL: {best_parseable:.2f}")
    print(f"Best Exact Match: {best_exact:.2f}")
    print(f"Best Latency: {best_latency:.3f}s")
    print("\nKey Findings:")
    print("  • Strict prompting improved parseable rate from 93% → 100%")
    print("  • Fine-tuned FP16 strict achieves best balance (100% parseable, 15% exact, 0.43s)")
    print("  • INT8 strict has highest exact match (20%) but 5x slower latency")
    print("  • Fine-tuned models outperform commercial APIs on exact match")


if __name__ == "__main__":
    import os
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    print("Generating evaluation visualizations...")
    print("-" * 85)
    
    plot_accuracy_comparison()
    plot_3d_bar_comparison()
    plot_3d_scatter()
    plot_accuracy_vs_latency()
    generate_summary_table()
    
    print("-" * 85)
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files in images/ directory:")
    print("  • results_comparison.png - 2D parseable rate and exact match bars")
    print("  • model_comparison_3d.png - 3D multi-metric comparison bars (NEW)")
    print("  • accuracy_latency_3d.png - 3D trade-off scatter plot (NEW)")
    print("  • accuracy_vs_latency.png - 2D accuracy-latency scatter")
    print("\nNext steps:")
    print("  1. Review the generated images")
    print("  2. Add workflow_architecture_diagram.svg to images/")
    print("  3. Commit all images to GitHub")
