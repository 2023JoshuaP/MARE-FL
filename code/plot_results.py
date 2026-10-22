import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    csv_file = "../results/summary_fashion.csv"
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(16, 6), dpi=300)
    palette = sns.color_palette("husl", len(df))

    # Bar Chart (Accuracy)
    ax1 = plt.subplot(1, 2, 1)
    bars = ax1.bar(df['strategy'], df['final_accuracy'], color=palette, edgecolor='black', linewidth=1.2)
    ax1.set_title('Final Accuracy by Strategy (Fashion-MNIST)', pad=15, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_xlabel('Strategy', fontweight='bold')
    ax1.set_ylim(0.70, 0.85)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

    # Scatter Plot (Latency vs Accuracy)
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=df, 
        x='mean_round_latency_s', 
        y='final_accuracy', 
        hue='strategy',
        palette=palette,
        s=200,
        ax=ax2,
        edgecolor='black'
    )
    
    for i in range(df.shape[0]):
        ax2.text(df['mean_round_latency_s'][i] + 0.01, 
                 df['final_accuracy'][i], 
                 df['strategy'][i], 
                 horizontalalignment='left', 
                 size='small', color='black', weight='semibold')

    ax2.set_title('Trade-off: Accuracy vs. Latency (Pareto)', pad=15, fontweight='bold')
    ax2.set_ylabel('Final Accuracy (Higher is better)', fontweight='bold')
    ax2.set_xlabel('Mean Round Latency (Seconds, Lower is better)', fontweight='bold')
    ax2.get_legend().remove()

    plt.tight_layout()
    output_img = "../results/plot_fashion.png"
    plt.savefig(output_img, bbox_inches='tight')
    print(f"Saved plot to {output_img}")

if __name__ == "__main__":
    main()

