#!/usr/bin/env python3
"""
Visualize aiperf benchmark results from profile_export_aiperf.json files.

Usage:
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks QWen3"
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks" --percentile p50 p95 p99
    python visualize.py --artifacts-dir ./artifacts --prefix qwen3 --label "QWen3"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Try to import seaborn for better styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def find_json_files(artifacts_dir: str, prefix: str) -> List[Path]:
    """Find all profile_export_aiperf.json files matching the prefix."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        print(f"Error: Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)
    
    json_files = []
    for subdir in sorted(artifacts_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith(prefix):
            json_file = subdir / "profile_export_aiperf.json"
            if json_file.exists():
                json_files.append(json_file)
    
    return json_files


def load_benchmark_data(json_files: List[Path]) -> List[Dict[str, Any]]:
    """Load benchmark data from JSON files."""
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            benchmark = json.load(f)
            # Extract concurrency from input_config
            concurrency = benchmark.get('input_config', {}).get('loadgen', {}).get('concurrency', 0)
            benchmark['_concurrency'] = concurrency
            benchmark['_file'] = str(json_file)
            data.append(benchmark)
    
    # Sort by concurrency
    data.sort(key=lambda x: x['_concurrency'])
    return data


# Fixed marker mapping for percentiles (consistent across all plots)
# All percentiles use the same color but different markers
PERCENTILE_MARKERS = {
    'p1':  'v',   # triangle down
    'p5':  '<',   # triangle left
    'p10': '>',   # triangle right
    'p25': 'p',   # pentagon
    'p50': 'o',   # circle
    'p75': 's',   # square
    'p90': '^',   # triangle up
    'p95': 'D',   # diamond
    'p99': 'h',   # hexagon
    'avg': '*',   # star
    'min': '+',   # plus
    'max': 'x',   # x
}

# Default line color for all percentiles
DEFAULT_COLOR = '#1f77b4'  # blue
DEFAULT_MARKER = 'o'


def get_percentile_style(percentile: str) -> Dict[str, str]:
    """Get consistent color and marker for a percentile."""
    return {
        'color': DEFAULT_COLOR,
        'marker': PERCENTILE_MARKERS.get(percentile, DEFAULT_MARKER)
    }


def get_metric_display_name(metric: str) -> str:
    """Get human-readable display name for a metric."""
    display_names = {
        'request_latency': 'Request Latency',
        'request_throughput': 'Request Throughput',
        'output_token_throughput': 'Output Token Throughput',
        'output_sequence_length': 'Output Sequence Length',
        'input_sequence_length': 'Input Sequence Length',
        'output_token_count': 'Output Token Count',
        'usage_prompt_tokens': 'Usage Prompt Tokens',
        'usage_completion_tokens': 'Usage Completion Tokens',
        'usage_total_tokens': 'Usage Total Tokens',
    }
    return display_names.get(metric, metric.replace('_', ' ').title())


def get_metric_unit(data: List[Dict[str, Any]], metric: str) -> str:
    """Get the unit for a metric from the data."""
    if data and metric in data[0]:
        return data[0][metric].get('unit', '')
    return ''


def plot_metric(
    data: List[Dict[str, Any]],
    label: str,
    metric: str,
    percentiles: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot metric percentiles across concurrency levels."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    concurrencies = [d['_concurrency'] for d in data]
    
    # Get metric display name and unit
    metric_name = get_metric_display_name(metric)
    metric_unit = get_metric_unit(data, metric)
    
    # Plot each percentile with consistent colors
    for percentile in percentiles:
        values = []
        for d in data:
            metric_data = d.get(metric, {})
            value = metric_data.get(percentile, metric_data.get('avg', 0))
            values.append(value)
        
        style = get_percentile_style(percentile)
        linestyle = ':' if percentile == 'p50' else '-'
        ax.plot(concurrencies, values, marker=style['marker'], linewidth=2, markersize=8,
                label=f'{percentile.upper()}', color=style['color'], linestyle=linestyle)
    
    ax.set_xlabel('Concurrency', fontsize=12)
    ylabel = f'{metric_name} ({metric_unit})' if metric_unit else metric_name
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{metric_name} vs Concurrency\n{label}', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all concurrency values
    ax.set_xticks(concurrencies)
    ax.set_xticklabels(concurrencies)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


# Keep for backward compatibility
def plot_request_latency(
    data: List[Dict[str, Any]],
    label: str,
    percentiles: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot request latency percentiles across concurrency levels."""
    plot_metric(data, label, 'request_latency', percentiles, output_file, show_plot)


def plot_throughput(
    data: List[Dict[str, Any]],
    label: str,
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot throughput metrics across concurrency levels."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    concurrencies = [d['_concurrency'] for d in data]
    
    # Request throughput
    request_throughput = [d.get('request_throughput', {}).get('avg', 0) for d in data]
    ax1.plot(concurrencies, request_throughput, marker='o', linewidth=2, markersize=8, color='tab:blue')
    ax1.set_xlabel('Concurrency', fontsize=12)
    ax1.set_ylabel('Requests/sec', fontsize=12)
    ax1.set_title('Request Throughput', fontsize=14)
    ax1.set_xticks(concurrencies)
    ax1.grid(True, alpha=0.3)
    
    # Token throughput
    token_throughput = [d.get('output_token_throughput', {}).get('avg', 0) for d in data]
    ax2.plot(concurrencies, token_throughput, marker='s', linewidth=2, markersize=8, color='tab:green')
    ax2.set_xlabel('Concurrency', fontsize=12)
    ax2.set_ylabel('Tokens/sec', fontsize=12)
    ax2.set_title('Output Token Throughput', fontsize=14)
    ax2.set_xticks(concurrencies)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Throughput vs Concurrency - {label}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def plot_combined(
    data: List[Dict[str, Any]],
    label: str,
    percentiles: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot combined latency and throughput metrics."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.style.use('ggplot')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    concurrencies = [d['_concurrency'] for d in data]
    
    # 1. Request Latency Percentiles
    ax1 = axes[0, 0]
    
    for percentile in percentiles:
        values = [d.get('request_latency', {}).get(percentile, 0) for d in data]
        style = get_percentile_style(percentile)
        linestyle = ':' if percentile == 'p50' else '-'
        ax1.plot(concurrencies, values, marker=style['marker'], 
                 linewidth=2, markersize=8, label=percentile.upper(), color=style['color'], linestyle=linestyle)
    
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Request Latency Percentiles')
    ax1.legend()
    ax1.set_xticks(concurrencies)
    ax1.grid(True, alpha=0.3)
    
    # 2. Request Throughput
    ax2 = axes[0, 1]
    request_throughput = [d.get('request_throughput', {}).get('avg', 0) for d in data]
    ax2.bar(range(len(concurrencies)), request_throughput, color='tab:blue', alpha=0.7)
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Requests/sec')
    ax2.set_title('Request Throughput')
    ax2.set_xticks(range(len(concurrencies)))
    ax2.set_xticklabels(concurrencies)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Token Throughput
    ax3 = axes[1, 0]
    token_throughput = [d.get('output_token_throughput', {}).get('avg', 0) for d in data]
    ax3.bar(range(len(concurrencies)), token_throughput, color='tab:green', alpha=0.7)
    ax3.set_xlabel('Concurrency')
    ax3.set_ylabel('Tokens/sec')
    ax3.set_title('Output Token Throughput')
    ax3.set_xticks(range(len(concurrencies)))
    ax3.set_xticklabels(concurrencies)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Latency Distribution (box-like using percentiles)
    ax4 = axes[1, 1]
    x = range(len(concurrencies))
    
    # Get percentile values
    p50 = [d.get('request_latency', {}).get('p50', 0) for d in data]
    p25 = [d.get('request_latency', {}).get('p25', 0) for d in data]
    p75 = [d.get('request_latency', {}).get('p75', 0) for d in data]
    p95 = [d.get('request_latency', {}).get('p95', 0) for d in data]
    p5 = [d.get('request_latency', {}).get('p5', 0) for d in data]
    
    # Plot as error bars
    ax4.errorbar(x, p50, yerr=[np.array(p50) - np.array(p25), np.array(p75) - np.array(p50)],
                 fmt='o', capsize=5, capthick=2, color='tab:blue', label='P25-P50-P75')
    ax4.scatter(x, p95, marker='^', s=50, color='tab:red', label='P95', zorder=5)
    ax4.scatter(x, p5, marker='v', s=50, color='tab:green', label='P5', zorder=5)
    
    ax4.set_xlabel('Concurrency')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Latency Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(concurrencies)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Benchmark Results - {label}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def print_summary(data: List[Dict[str, Any]], label: str):
    """Print a summary table of the benchmark results."""
    print(f"\n{'='*80}")
    print(f"Benchmark Summary: {label}")
    print(f"{'='*80}")
    
    headers = ['Concurrency', 'Req/s', 'Tokens/s', 'Latency Avg (ms)', 'P50 (ms)', 'P95 (ms)', 'P99 (ms)']
    print(f"{headers[0]:<12} {headers[1]:<10} {headers[2]:<12} {headers[3]:<16} {headers[4]:<12} {headers[5]:<12} {headers[6]:<12}")
    print('-' * 80)
    
    for d in data:
        concurrency = d['_concurrency']
        req_throughput = d.get('request_throughput', {}).get('avg', 0)
        token_throughput = d.get('output_token_throughput', {}).get('avg', 0)
        latency = d.get('request_latency', {})
        
        print(f"{concurrency:<12} {req_throughput:<10.2f} {token_throughput:<12.2f} "
              f"{latency.get('avg', 0):<16.2f} {latency.get('p50', 0):<12.2f} "
              f"{latency.get('p95', 0):<12.2f} {latency.get('p99', 0):<12.2f}")
    
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize aiperf benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks QWen3"
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks" --percentile p50 p95 p99
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks" --metric output_sequence_length
    python visualize.py --artifacts-dir ./artifacts --prefix qwen3 --label "QWen3" --output results.png
        """
    )
    
    parser.add_argument('--prefix', type=str, required=True,
                        help='Prefix to match artifact directories (e.g., "accounts_pyroworks")')
    parser.add_argument('--label', type=str, required=True,
                        help='Label for the plot legend/title')
    parser.add_argument('--artifacts-dir', type=str, default='./artifacts',
                        help='Directory containing benchmark artifacts (default: ./artifacts)')
    parser.add_argument('--percentile', type=str, nargs='+', default=['p50', 'p95'],
                        help='Percentiles to plot (default: p50 p95)')
    parser.add_argument('--metric', type=str, default='request_latency',
                        help='Metric to plot (default: request_latency). Available: request_latency, '
                             'output_sequence_length, input_sequence_length, usage_prompt_tokens, '
                             'usage_completion_tokens, usage_total_tokens')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for the plot (optional)')
    parser.add_argument('--plot-type', type=str, choices=['latency', 'throughput', 'combined', 'all'],
                        default='latency',
                        help='Type of plot to generate (default: latency)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (only save to file)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary table')
    
    args = parser.parse_args()
    
    # Find and load JSON files
    print(f"Searching for benchmarks with prefix: {args.prefix}")
    json_files = find_json_files(args.artifacts_dir, args.prefix)
    
    if not json_files:
        print(f"Error: No benchmark files found with prefix '{args.prefix}' in {args.artifacts_dir}")
        sys.exit(1)
    
    print(f"Found {len(json_files)} benchmark file(s):")
    for f in json_files:
        print(f"  - {f}")
    
    # Load benchmark data
    data = load_benchmark_data(json_files)
    
    # Print summary if requested
    if args.summary:
        print_summary(data, args.label)
    
    # Generate plots
    show_plot = not args.no_show
    
    if args.plot_type == 'latency':
        plot_metric(data, args.label, args.metric, args.percentile, args.output, show_plot)
    elif args.plot_type == 'throughput':
        output = args.output.replace('.png', '_throughput.png') if args.output else None
        plot_throughput(data, args.label, output, show_plot)
    elif args.plot_type == 'combined':
        plot_combined(data, args.label, args.percentile, args.output, show_plot)
    elif args.plot_type == 'all':
        base_output = args.output.replace('.png', '') if args.output else None
        plot_metric(data, args.label, args.metric, args.percentile, 
                   f"{base_output}_latency.png" if base_output else None, show_plot)
        plot_throughput(data, args.label,
                       f"{base_output}_throughput.png" if base_output else None, show_plot)
        plot_combined(data, args.label, args.percentile,
                     f"{base_output}_combined.png" if base_output else None, show_plot)


if __name__ == '__main__':
    main()
