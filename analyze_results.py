import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

OUTPUT_DIR = 'automcp_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_process_data(filepath='results.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['version'])
    df = df.sort_values('date')
    
    df['efficiency_score'] = df['generation_score_ratio'] / df['generation_time_seconds']
    df['tools_per_second'] = df['tools_generated'] / df['generation_time_seconds']
    df['resource_efficiency'] = df['tools_generated'] / (df['memory_usage_mb'] * df['cpu_usage_percent'])
    df['total_time'] = df['generation_time_seconds'] + df['downtime_seconds']
    
    return df

def create_performance_trends(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['generation_time_seconds'], marker='o', linewidth=2, markersize=6)
    plt.title('Generation Time Evolution', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_generation_time_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['efficiency_score'], marker='s', color='green', linewidth=2, markersize=6)
    plt.title('Generation Efficiency', fontsize=14, fontweight='bold')
    plt.ylabel('Efficiency Score')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_generation_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['tools_generated'], marker='^', color='orange', linewidth=2, markersize=6)
    plt.title('Tools Generated Growth', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Tools')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_tools_generated_growth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['downtime_seconds'], marker='d', color='red', linewidth=2, markersize=6)
    plt.title('System Downtime', fontsize=14, fontweight='bold')
    plt.ylabel('Downtime (seconds)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_system_downtime.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_resource_utilization(df):
    plt.figure(figsize=(10, 6))
    plt.fill_between(df['date'], df['memory_usage_mb'], alpha=0.7, color='blue')
    plt.plot(df['date'], df['memory_usage_mb'], marker='o', color='darkblue', linewidth=2, markersize=6)
    plt.title('Memory Usage Over Time', fontsize=14, fontweight='bold')
    plt.ylabel('Memory (MB)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_memory_usage_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(df['date'], df['cpu_usage_percent'], alpha=0.7, color='red')
    plt.plot(df['date'], df['cpu_usage_percent'], marker='s', color='darkred', linewidth=2, markersize=6)
    plt.title('CPU Usage Pattern', fontsize=14, fontweight='bold')
    plt.ylabel('CPU Usage (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_cpu_usage_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['resource_efficiency'], marker='^', color='green', linewidth=2, markersize=6)
    plt.title('Resource Efficiency', fontsize=14, fontweight='bold')
    plt.ylabel('Efficiency Ratio')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_resource_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['memory_usage_mb'], df['cpu_usage_percent'], 
                         c=df['tools_generated'], cmap='viridis', s=100, alpha=0.7)
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('CPU Usage (%)')
    plt.title('Memory vs CPU Usage', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Tools Generated')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_memory_vs_cpu_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_scale_analysis(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['generation_score_ratio'], marker='o', color='purple', linewidth=2, markersize=6)
    plt.axhline(y=df['generation_score_ratio'].mean(), color='red', linestyle='--', 
               label=f'Average: {df["generation_score_ratio"].mean():.2f}')
    plt.title('Generation Quality Over Time', fontsize=14, fontweight='bold')
    plt.ylabel('Quality Score Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_generation_quality_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['tools_per_second'], marker='s', color='orange', linewidth=2, markersize=6)
    plt.title('Generation Speed', fontsize=14, fontweight='bold')
    plt.ylabel('Tools/Second')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_generation_speed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['tools_generated'], df['generation_score_ratio'], 
                         c=df['generation_time_seconds'], cmap='plasma', s=100, alpha=0.7)
    plt.xlabel('Tools Generated')
    plt.ylabel('Generation Quality')
    plt.title('Quality vs Scale', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Generation Time (s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_quality_vs_scale_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['generation_time_seconds'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['generation_time_seconds'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["generation_time_seconds"].mean():.2f}s')
    plt.xlabel('Generation Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Generation Time Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/12_generation_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_additional_charts(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['date'], df['generation_time_seconds'], 'b-o', label='Generation Time', linewidth=2, markersize=6)
    line2 = ax2.plot(df['date'], df['tools_generated'], 'r-s', label='Tools Generated', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Generation Time (seconds)', color='b')
    ax2.set_ylabel('Tools Generated', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Performance vs Scale Over Time', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/13_performance_vs_scale.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['memory_usage_mb'], df['generation_time_seconds'], 
               c=df['cpu_usage_percent'], cmap='coolwarm', s=100, alpha=0.7)
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Generation Time (seconds)')
    plt.title('Memory vs Generation Time', fontsize=14, fontweight='bold')
    plt.colorbar(label='CPU Usage (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/14_memory_vs_time_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(df['date'], df['efficiency_score'], alpha=0.5, color='green')
    plt.plot(df['date'], df['efficiency_score'], marker='o', color='darkgreen', linewidth=2, markersize=6)
    plt.title('Efficiency Evolution', fontsize=14, fontweight='bold')
    plt.ylabel('Efficiency Score')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/15_efficiency_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    stability = df['generation_time_seconds'].rolling(window=3).std() / df['generation_time_seconds'].rolling(window=3).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], stability, marker='d', color='navy', linewidth=2, markersize=6)
    plt.title('System Stability', fontsize=14, fontweight='bold')
    plt.ylabel('Stability (Lower is Better)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/16_system_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_unique_combinations(df):
    plt.figure(figsize=(12, 8))
    bubble_sizes = df['generation_time_seconds'] * 20
    scatter = plt.scatter(df['tools_generated'], df['efficiency_score'], 
                         s=bubble_sizes, c=df['memory_usage_mb'], 
                         cmap='plasma', alpha=0.6, edgecolors='black')
    plt.xlabel('Tools Generated')
    plt.ylabel('Efficiency Score')
    plt.title('Efficiency Bubble Chart (Size: Generation Time, Color: Memory)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Memory Usage (MB)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/17_efficiency_bubble_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.scatter(df['tools_generated'], df['memory_usage_mb'], c='blue', alpha=0.6)
    ax1.set_xlabel('Tools Generated')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Scale vs Memory Usage')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(df['tools_generated'], df['cpu_usage_percent'], c='red', alpha=0.6)
    ax2.set_xlabel('Tools Generated')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.set_title('Scale vs CPU Usage')
    ax2.grid(True, alpha=0.3)
    
    ax3.scatter(df['efficiency_score'], df['downtime_seconds'], c='green', alpha=0.6)
    ax3.set_xlabel('Efficiency Score')
    ax3.set_ylabel('Downtime (seconds)')
    ax3.set_title('Efficiency vs Downtime')
    ax3.grid(True, alpha=0.3)
    
    ax4.scatter(df['resource_efficiency'], df['tools_per_second'], c='purple', alpha=0.6)
    ax4.set_xlabel('Resource Efficiency')
    ax4.set_ylabel('Tools per Second')
    ax4.set_title('Resource vs Speed Efficiency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/18_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    df['year'] = df['date'].dt.year
    yearly_avg = df.groupby('year').agg({
        'generation_time_seconds': 'mean',
        'tools_generated': 'mean',
        'memory_usage_mb': 'mean',
        'cpu_usage_percent': 'mean',
        'efficiency_score': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.15
    x = np.arange(len(yearly_avg))
    
    ax.bar(x - 2*width, yearly_avg['generation_time_seconds'], width, label='Gen Time (s)', alpha=0.8)
    ax.bar(x - width, yearly_avg['tools_generated']/2, width, label='Tools Generated/2', alpha=0.8)
    ax.bar(x, yearly_avg['memory_usage_mb'], width, label='Memory (MB)', alpha=0.8)
    ax.bar(x + width, yearly_avg['cpu_usage_percent']*10, width, label='CPU % x10', alpha=0.8)
    ax.bar(x + 2*width, yearly_avg['efficiency_score']*100, width, label='Efficiency x100', alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Values')
    ax.set_title('Yearly Performance Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_avg['year'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/19_yearly_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    metrics = ['generation_time_seconds', 'memory_usage_mb', 'cpu_usage_percent', 
              'tools_generated', 'efficiency_score', 'downtime_seconds']
    correlation_matrix = df[metrics].corr()
    
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.xticks(range(len(metrics)), [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    plt.yticks(range(len(metrics)), [m.replace('_', ' ').title() for m in metrics])
    plt.title('Performance Metrics Correlation Heatmap', fontsize=14, fontweight='bold')
    
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/20_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    df['performance_tier'] = pd.cut(df['generation_time_seconds'], 
                                   bins=3, labels=['Fast', 'Medium', 'Slow'])
    df['scale_tier'] = pd.cut(df['tools_generated'], 
                             bins=3, labels=['Small', 'Medium', 'Large'])
    
    tier_analysis = df.groupby(['performance_tier', 'scale_tier'], observed=False).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    tier_analysis.plot(kind='bar', ax=ax, color=['lightblue', 'orange', 'lightgreen'])
    plt.title('Performance vs Scale Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Performance Tier')
    plt.ylabel('Count')
    plt.legend(title='Scale Tier')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/21_tier_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    numeric_cols = ['generation_time_seconds', 'efficiency_score', 'memory_usage_mb', 'tools_generated']
    rolling_metrics = df.set_index('date')[numeric_cols].rolling(window=5, min_periods=1).mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].plot(rolling_metrics.index, rolling_metrics['generation_time_seconds'], 
                   'b-', linewidth=2, label='5-period MA')
    axes[0,0].plot(df['date'], df['generation_time_seconds'], 'b--', alpha=0.5, label='Actual')
    axes[0,0].set_title('Generation Time Trend Analysis')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(rolling_metrics.index, rolling_metrics['efficiency_score'], 
                   'g-', linewidth=2, label='5-period MA')
    axes[0,1].plot(df['date'], df['efficiency_score'], 'g--', alpha=0.5, label='Actual')
    axes[0,1].set_title('Efficiency Trend Analysis')
    axes[0,1].set_ylabel('Efficiency Score')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(rolling_metrics.index, rolling_metrics['memory_usage_mb'], 
                   'r-', linewidth=2, label='5-period MA')
    axes[1,0].plot(df['date'], df['memory_usage_mb'], 'r--', alpha=0.5, label='Actual')
    axes[1,0].set_title('Memory Usage Trend Analysis')
    axes[1,0].set_ylabel('Memory (MB)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(rolling_metrics.index, rolling_metrics['tools_generated'], 
                   'purple', linewidth=2, label='5-period MA')
    axes[1,1].plot(df['date'], df['tools_generated'], color='purple', linestyle='--', alpha=0.5, label='Actual')
    axes[1,1].set_title('Tools Generated Trend Analysis')
    axes[1,1].set_ylabel('Tools Generated')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/22_trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    df['efficiency_change'] = df['efficiency_score'].pct_change()
    df['tools_change'] = df['tools_generated'].pct_change()
    df['time_change'] = df['generation_time_seconds'].pct_change()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(df['tools_change'], df['efficiency_change'], 
               c=df['time_change'], cmap='RdYlBu', s=100, alpha=0.7)
    plt.xlabel('Tools Generated Change (%)')
    plt.ylabel('Efficiency Change (%)')
    plt.title('Performance Change Dynamics', fontsize=14, fontweight='bold')
    plt.colorbar(label='Time Change (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/23_change_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()



def create_summary_statistics(df):
    print("AutoMCP Performance Analysis Summary")
    print("=" * 40)
    
    print(f"\nDataset Overview:")
    print(f"Time period: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"Total evaluations: {len(df)}")
    print(f"Success rate: {df['generation_success'].mean()*100:.1f}%")
    
    print(f"\nPerformance Metrics:")
    print(f"Avg generation time: {df['generation_time_seconds'].mean():.2f}s (±{df['generation_time_seconds'].std():.2f})")
    print(f"Fastest generation: {df['generation_time_seconds'].min():.2f}s")
    print(f"Slowest generation: {df['generation_time_seconds'].max():.2f}s")
    print(f"Avg downtime: {df['downtime_seconds'].mean():.3f}s")
    
    print(f"\nResource Utilization:")
    print(f"Avg memory usage: {df['memory_usage_mb'].mean():.1f} MB")
    print(f"Avg CPU usage: {df['cpu_usage_percent'].mean():.1f}%")
    print(f"Resource efficiency: {df['resource_efficiency'].mean():.3f} tools per resource unit")
    
    print(f"\nQuality & Scale:")
    print(f"Avg quality score: {df['generation_score_ratio'].mean():.2f}")
    print(f"Tools generated range: {df['tools_generated'].min()}-{df['tools_generated'].max()}")
    print(f"Total tools generated: {df['tools_generated'].sum()}")
    print(f"Avg generation efficiency: {df['efficiency_score'].mean():.4f} quality/second")
    
    latest_performance = df['generation_time_seconds'].iloc[-5:].mean()
    early_performance = df['generation_time_seconds'].iloc[:5].mean()
    performance_change = ((latest_performance - early_performance) / early_performance) * 100
    
    print(f"\nTrends (Latest 5 vs First 5 evaluations):")
    print(f"Performance change: {performance_change:+.1f}%")
    print(f"Tools growth: {df['tools_generated'].iloc[-1]} vs {df['tools_generated'].iloc[0]} (+{((df['tools_generated'].iloc[-1] - df['tools_generated'].iloc[0]) / df['tools_generated'].iloc[0]) * 100:.0f}%)")
    
    best_efficiency = df.loc[df['efficiency_score'].idxmax()]
    print(f"\nNotable Achievements:")
    print(f"Best efficiency: {best_efficiency['efficiency_score']:.4f} on {best_efficiency['date'].strftime('%Y-%m')}")
    print(f"Most tools generated: {df['tools_generated'].max()} tools")
    print(f"Lowest resource usage: {df['memory_usage_mb'].min():.1f} MB memory, {df['cpu_usage_percent'].min():.1f}% CPU")

def main():
    print("Loading evaluation data...")
    df = load_and_process_data()
    
    print(f"Creating visualizations in '{OUTPUT_DIR}' folder...")
    
    print("Creating performance trend charts...")
    create_performance_trends(df)
    
    print("Creating resource utilization charts...")
    create_resource_utilization(df)
    
    print("Creating quality and scale analysis charts...")
    create_quality_scale_analysis(df)
    
    print("Creating additional insight charts...")
    create_additional_charts(df)
    
    print("Creating unique combination analyses...")
    create_unique_combinations(df)
    
    create_summary_statistics(df)
    
    print(f"\nAnalysis complete! All PNG charts saved in '{OUTPUT_DIR}/' folder:")
    print("01_generation_time_evolution.png")
    print("02_generation_efficiency.png")
    print("03_tools_generated_growth.png")
    print("04_system_downtime.png")
    print("05_memory_usage_over_time.png")
    print("06_cpu_usage_pattern.png")
    print("07_resource_efficiency.png")
    print("08_memory_vs_cpu_scatter.png")
    print("09_generation_quality_over_time.png")
    print("10_generation_speed.png")
    print("11_quality_vs_scale_correlation.png")
    print("12_generation_time_distribution.png")
    print("13_performance_vs_scale.png")
    print("14_memory_vs_time_correlation.png")
    print("15_efficiency_evolution.png")
    print("16_system_stability.png")
    print("17_efficiency_bubble_chart.png")
    print("18_correlation_matrix.png")
    print("19_yearly_comparison.png")
    print("20_correlation_heatmap.png")
    print("21_tier_distribution.png")
    print("22_trend_analysis.png")
    print("23_change_dynamics.png")

if __name__ == "__main__":
    main()