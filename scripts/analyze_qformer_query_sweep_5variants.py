#!/usr/bin/env python3
"""
Analyze Q-Former Query Count Sweep Results (5 variants: 8, 16, 32, 64, 128)
Creates token-quality frontier visualization and summary table
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_dir, query_counts):
    """Load evaluation metrics for all query count variants"""
    results = {}
    
    for num_queries in query_counts:
        projector_name = f"qformer-d3-l{num_queries}"
        metrics_file = os.path.join(
            results_dir, 
            f"coco_metrics_val2014_{projector_name}.json"
        )
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results[num_queries] = metrics
                print(f"✓ Loaded results for {num_queries} queries")
        else:
            print(f"✗ Missing results for {num_queries} queries: {metrics_file}")
    
    return results

def create_summary_table(results, output_dir):
    """Create markdown summary table"""
    
    # Sort by query count
    sorted_queries = sorted(results.keys())
    
    # Create table
    table = "# Q-Former Query Count Sweep Results (5 Variants)\n\n"
    table += "| Queries | Tokens | Compression | CIDEr | BLEU-4 | METEOR | ROUGE-L | SPICE |\n"
    table += "|---------|--------|-------------|-------|--------|--------|---------|-------|\n"
    
    for num_queries in sorted_queries:
        metrics = results[num_queries]
        compression_ratio = 576 / num_queries  # From 24×24 = 576 vision tokens
        
        table += f"| {num_queries:3d} | {num_queries:3d} | {compression_ratio:5.1f}× | "
        table += f"{metrics.get('CIDEr', 0):.3f} | "
        table += f"{metrics.get('Bleu_4', 0):.3f} | "
        table += f"{metrics.get('METEOR', 0):.3f} | "
        table += f"{metrics.get('ROUGE_L', 0):.3f} | "
        table += f"{metrics.get('SPICE', 0):.3f} |\n"
    
    # Save table
    table_file = os.path.join(output_dir, "query_sweep_summary.md")
    with open(table_file, 'w') as f:
        f.write(table)
    
    print(f"\n✓ Summary table saved to: {table_file}")
    return table

def plot_token_quality_frontier(results, output_dir):
    """Create token-quality frontier visualization"""
    
    # Extract data
    query_counts = sorted(results.keys())
    cider_scores = [results[q].get('CIDEr', 0) for q in query_counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot main curve
    ax.plot(query_counts, cider_scores, 'o-', linewidth=2, markersize=8, 
            color='#2E86AB', label='CIDEr Score')
    
    # Highlight optimal range (32-64 queries expected)
    optimal_range = [q for q in query_counts if 32 <= q <= 64]
    optimal_scores = [results[q].get('CIDEr', 0) for q in optimal_range]
    if optimal_range:
        ax.scatter(optimal_range, optimal_scores, s=200, c='#A23B72', 
                  marker='*', zorder=5, label='Expected Optimal Range')
    
    # Formatting
    ax.set_xlabel('Number of Query Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('CIDEr Score', fontsize=12, fontweight='bold')
    ax.set_title('Q-Former Token-Quality Frontier\n(COCO Captioning, 5 Variants)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Add compression ratio on secondary x-axis
    ax2 = ax.twiny()
    compression_ratios = [576/q for q in query_counts]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(query_counts)
    ax2.set_xticklabels([f"{576/q:.1f}×" for q in query_counts])
    ax2.set_xlabel('Compression Ratio (from 576 tokens)', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = os.path.join(output_dir, "token_quality_frontier.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    
    plt.close()

def analyze_optimal_range(results):
    """Identify optimal query count range"""
    
    query_counts = sorted(results.keys())
    cider_scores = [results[q].get('CIDEr', 0) for q in query_counts]
    
    # Find peak
    max_idx = np.argmax(cider_scores)
    optimal_queries = query_counts[max_idx]
    max_cider = cider_scores[max_idx]
    
    # Find plateau range (within 1% of max)
    threshold = max_cider * 0.99
    plateau_queries = [q for q, s in zip(query_counts, cider_scores) if s >= threshold]
    
    print("\n" + "="*50)
    print("OPTIMAL RANGE ANALYSIS")
    print("="*50)
    print(f"Peak Performance: {optimal_queries} queries (CIDEr: {max_cider:.3f})")
    if len(plateau_queries) > 1:
        print(f"Plateau Range (≥99% of peak): {min(plateau_queries)}-{max(plateau_queries)} queries")
    else:
        print(f"Plateau Range (≥99% of peak): {plateau_queries[0]} queries")
    print(f"Compression at Peak: {576/optimal_queries:.1f}× reduction")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze Q-Former Query Sweep Results (5 variants)")
    parser.add_argument("--results_dir", type=str, default="./results/qformer_query_sweep",
                       help="Directory containing evaluation results")
    parser.add_argument("--query_counts", nargs="+", type=int, 
                       default=[8, 16, 32, 64, 128],
                       help="Query counts to analyze")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (defaults to results_dir)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print("Q-FORMER QUERY COUNT SWEEP ANALYSIS (5 VARIANTS)")
    print("="*50)
    
    # Load results
    results = load_results(args.results_dir, args.query_counts)
    
    if not results:
        print("\n✗ No results found! Please run evaluation first.")
        return
    
    # Create summary table
    create_summary_table(results, args.output_dir)
    
    # Create visualization
    plot_token_quality_frontier(results, args.output_dir)
    
    # Analyze optimal range
    analyze_optimal_range(results)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()