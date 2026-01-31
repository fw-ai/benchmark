#!/usr/bin/env python3
"""
Visualize aiperf benchmark results from profile_export_aiperf.json files.

Usage:
    # Single prefix
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks QWen3"
    
    # Multiple prefixes for comparison
    python visualize.py --prefix fw_pyroworks trtllm_pyroworks --label "Fireworks" "TRT-LLM"
    
    # With percentiles
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks" --percentile p50 p95 p99
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

# Colors for multiple prefixes
PREFIX_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
]


def get_percentile_style(percentile: str, color_idx: int = 0) -> Dict[str, str]:
    """Get consistent color and marker for a percentile."""
    return {
        'color': PREFIX_COLORS[color_idx % len(PREFIX_COLORS)],
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
    datasets: List[List[Dict[str, Any]]],
    labels: List[str],
    metric: str,
    percentiles: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot metric percentiles across concurrency levels for multiple datasets."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get metric display name and unit from first dataset
    metric_name = get_metric_display_name(metric)
    metric_unit = get_metric_unit(datasets[0], metric) if datasets else ''
    
    # Determine primary metric (last percentile in the list)
    primary_percentile = percentiles[-1] if percentiles else None
    
    # Collect all concurrencies for x-axis
    all_concurrencies = set()
    for data in datasets:
        all_concurrencies.update([d['_concurrency'] for d in data])
    all_concurrencies = sorted(all_concurrencies)
    
    # Plot each dataset with different colors
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        concurrencies = [d['_concurrency'] for d in data]
        
        # Plot each percentile
        for percentile in percentiles:
            values = []
            for d in data:
                metric_data = d.get(metric, {})
                value = metric_data.get(percentile, metric_data.get('avg', 0))
                values.append(value)
            
            style = get_percentile_style(percentile, data_idx)
            linestyle = ':' if percentile == 'p50' else '-'
            
            # Include label in legend only for multi-dataset or single percentile
            if len(datasets) > 1:
                plot_label = f'{label} {percentile.upper()}'
            else:
                plot_label = f'{percentile.upper()}'
            
            ax.plot(concurrencies, values, marker=style['marker'], linewidth=2, markersize=8,
                    label=plot_label, color=style['color'], linestyle=linestyle)
            
            # Add value annotations for primary metric
            if percentile == primary_percentile:
                # Offset annotations vertically based on dataset index to avoid overlap
                y_offset = 10 + (data_idx * 15)  # Stagger by 15 points per dataset
                for x, y in zip(concurrencies, values):
                    ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points', 
                               xytext=(0, y_offset), ha='center', fontsize=9, fontweight='bold',
                               color=style['color'])
    
    ax.set_xlabel('Concurrency', fontsize=12)
    ylabel = f'{metric_name} ({metric_unit})' if metric_unit else metric_name
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Title based on number of datasets
    if len(labels) == 1:
        ax.set_title(f'{metric_name} vs Concurrency\n{labels[0]}', fontsize=14)
    else:
        ax.set_title(f'{metric_name} vs Concurrency\nComparison: {", ".join(labels)}', fontsize=14)
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all concurrency values
    ax.set_xticks(all_concurrencies)
    ax.set_xticklabels(all_concurrencies)
    
    # Build info text for each prefix
    info_lines = []
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        color = PREFIX_COLORS[data_idx % len(PREFIX_COLORS)]
        avg_input = np.mean([d.get('input_sequence_length', {}).get('avg', 0) for d in data])
        avg_output = np.mean([d.get('output_sequence_length', {}).get('avg', 0) for d in data])
        avg_requests = np.mean([d.get('request_count', {}).get('avg', 0) for d in data])
        avg_failed = np.mean([len(d.get('error_summary', [])) for d in data])
        
        if len(datasets) > 1:
            info_lines.append(f'[{label}]')
        info_lines.append(f'In: {avg_input:.0f} / Out: {avg_output:.0f} tok')
        info_lines.append(f'Req: {avg_requests:.0f} / Fail: {avg_failed:.0f}')
        if data_idx < len(datasets) - 1:
            info_lines.append('─────────────')
    
    info_text = '\n'.join(info_lines)
    ax.text(1.02, 0.5, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for the text box
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


# Keep for backward compatibility
def plot_request_latency(
    datasets: List[List[Dict[str, Any]]],
    labels: List[str],
    percentiles: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot request latency percentiles across concurrency levels."""
    plot_metric(datasets, labels, 'request_latency', percentiles, output_file, show_plot)


def plot_throughput(
    datasets: List[List[Dict[str, Any]]],
    labels: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot throughput metrics across concurrency levels for multiple datasets."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect all concurrencies for x-axis
    all_concurrencies = set()
    for data in datasets:
        all_concurrencies.update([d['_concurrency'] for d in data])
    all_concurrencies = sorted(all_concurrencies)
    
    # Plot each dataset
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        concurrencies = [d['_concurrency'] for d in data]
        color = PREFIX_COLORS[data_idx % len(PREFIX_COLORS)]
        
        # Request throughput
        request_throughput = [d.get('request_throughput', {}).get('avg', 0) for d in data]
        ax1.plot(concurrencies, request_throughput, marker='o', linewidth=2, markersize=8, 
                 color=color, label=label)
        # Add value annotations
        y_offset = 10 + (data_idx * 15)
        for x, y in zip(concurrencies, request_throughput):
            ax1.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                        xytext=(0, y_offset), ha='center', fontsize=8, fontweight='bold', color=color)
        
        # Token throughput
        token_throughput = [d.get('output_token_throughput', {}).get('avg', 0) for d in data]
        ax2.plot(concurrencies, token_throughput, marker='s', linewidth=2, markersize=8, 
                 color=color, label=label)
        # Add value annotations
        for x, y in zip(concurrencies, token_throughput):
            ax2.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                        xytext=(0, y_offset), ha='center', fontsize=8, fontweight='bold', color=color)
    
    ax1.set_xlabel('Concurrency', fontsize=12)
    ax1.set_ylabel('Requests/sec', fontsize=12)
    ax1.set_title('Request Throughput', fontsize=14)
    ax1.set_xticks(all_concurrencies)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Concurrency', fontsize=12)
    ax2.set_ylabel('Tokens/sec', fontsize=12)
    ax2.set_title('Output Token Throughput', fontsize=14)
    ax2.set_xticks(all_concurrencies)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Build subtitle with per-prefix stats
    subtitle_parts = []
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        avg_input = np.mean([d.get('input_sequence_length', {}).get('avg', 0) for d in data])
        avg_output = np.mean([d.get('output_sequence_length', {}).get('avg', 0) for d in data])
        avg_requests = np.mean([d.get('request_count', {}).get('avg', 0) for d in data])
        avg_failed = np.mean([len(d.get('error_summary', [])) for d in data])
        if len(datasets) > 1:
            subtitle_parts.append(f'{label}: In={avg_input:.0f}/Out={avg_output:.0f}, Req={avg_requests:.0f}/Fail={avg_failed:.0f}')
        else:
            subtitle_parts.append(f'In: {avg_input:.0f}, Out: {avg_output:.0f} tok | Req: {avg_requests:.0f}, Fail: {avg_failed:.0f}')
    
    title_label = labels[0] if len(labels) == 1 else "Comparison"
    subtitle = subtitle_parts[0] if len(datasets) == 1 else ' | '.join(subtitle_parts)
    fig.suptitle(f'Throughput vs Concurrency - {title_label}\n{subtitle}', fontsize=12, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def plot_combined(
    datasets: List[List[Dict[str, Any]]],
    labels: List[str],
    percentiles: List[str],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """Plot combined latency and throughput metrics for multiple datasets."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.style.use('ggplot')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect all concurrencies
    all_concurrencies = set()
    for data in datasets:
        all_concurrencies.update([d['_concurrency'] for d in data])
    all_concurrencies = sorted(all_concurrencies)
    
    # 1. Request Latency Percentiles
    ax1 = axes[0, 0]
    primary_percentile = percentiles[-1] if percentiles else None
    
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        concurrencies = [d['_concurrency'] for d in data]
        for percentile in percentiles:
            values = [d.get('request_latency', {}).get(percentile, 0) for d in data]
            style = get_percentile_style(percentile, data_idx)
            linestyle = ':' if percentile == 'p50' else '-'
            plot_label = f'{label} {percentile.upper()}' if len(datasets) > 1 else percentile.upper()
            ax1.plot(concurrencies, values, marker=style['marker'], 
                     linewidth=2, markersize=8, label=plot_label, color=style['color'], linestyle=linestyle)
            # Add value annotations for primary percentile
            if percentile == primary_percentile:
                y_offset = 5 + (data_idx * 12)
                for x, y in zip(concurrencies, values):
                    ax1.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                                xytext=(0, y_offset), ha='center', fontsize=7, fontweight='bold',
                                color=style['color'])
    
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Request Latency Percentiles')
    ax1.legend(fontsize=8)
    ax1.set_xticks(all_concurrencies)
    ax1.grid(True, alpha=0.3)
    
    # 2. Request Throughput
    ax2 = axes[0, 1]
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        concurrencies = [d['_concurrency'] for d in data]
        request_throughput = [d.get('request_throughput', {}).get('avg', 0) for d in data]
        color = PREFIX_COLORS[data_idx % len(PREFIX_COLORS)]
        ax2.plot(concurrencies, request_throughput, marker='o', linewidth=2, markersize=8,
                 color=color, label=label)
        # Add value annotations
        y_offset = 5 + (data_idx * 12)
        for x, y in zip(concurrencies, request_throughput):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                        xytext=(0, y_offset), ha='center', fontsize=7, fontweight='bold', color=color)
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Requests/sec')
    ax2.set_title('Request Throughput')
    ax2.set_xticks(all_concurrencies)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Token Throughput
    ax3 = axes[1, 0]
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        concurrencies = [d['_concurrency'] for d in data]
        token_throughput = [d.get('output_token_throughput', {}).get('avg', 0) for d in data]
        color = PREFIX_COLORS[data_idx % len(PREFIX_COLORS)]
        ax3.plot(concurrencies, token_throughput, marker='s', linewidth=2, markersize=8,
                 color=color, label=label)
        # Add value annotations
        y_offset = 5 + (data_idx * 12)
        for x, y in zip(concurrencies, token_throughput):
            ax3.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                        xytext=(0, y_offset), ha='center', fontsize=7, fontweight='bold', color=color)
    ax3.set_xlabel('Concurrency')
    ax3.set_ylabel('Tokens/sec')
    ax3.set_title('Output Token Throughput')
    ax3.set_xticks(all_concurrencies)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Latency Distribution (box-like using percentiles)
    ax4 = axes[1, 1]
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        concurrencies = [d['_concurrency'] for d in data]
        x = concurrencies
        color = PREFIX_COLORS[data_idx % len(PREFIX_COLORS)]
        
        p50 = [d.get('request_latency', {}).get('p50', 0) for d in data]
        p25 = [d.get('request_latency', {}).get('p25', 0) for d in data]
        p75 = [d.get('request_latency', {}).get('p75', 0) for d in data]
        p95 = [d.get('request_latency', {}).get('p95', 0) for d in data]
        
        ax4.errorbar(x, p50, yerr=[np.array(p50) - np.array(p25), np.array(p75) - np.array(p50)],
                     fmt='o', capsize=5, capthick=2, color=color, label=f'{label} P25-P50-P75')
        ax4.scatter(x, p95, marker='^', s=50, color=color, zorder=5)
    
    ax4.set_xlabel('Concurrency')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Latency Distribution')
    ax4.set_xticks(all_concurrencies)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Build subtitle with per-prefix stats
    title_label = labels[0] if len(labels) == 1 else "Comparison"
    subtitle_parts = []
    for data_idx, (data, label) in enumerate(zip(datasets, labels)):
        avg_input = np.mean([d.get('input_sequence_length', {}).get('avg', 0) for d in data])
        avg_output = np.mean([d.get('output_sequence_length', {}).get('avg', 0) for d in data])
        avg_requests = np.mean([d.get('request_count', {}).get('avg', 0) for d in data])
        avg_failed = np.mean([len(d.get('error_summary', [])) for d in data])
        if len(datasets) > 1:
            subtitle_parts.append(f'{label}: In={avg_input:.0f}/Out={avg_output:.0f}, Req={avg_requests:.0f}/Fail={avg_failed:.0f}')
        else:
            subtitle_parts.append(f'In: {avg_input:.0f}, Out: {avg_output:.0f} tokens | Req: {avg_requests:.0f}, Fail: {avg_failed:.0f}')
    
    subtitle = ' | '.join(subtitle_parts) if len(datasets) == 1 else '\n'.join(subtitle_parts)
    fig.suptitle(f'Benchmark Results - {title_label}\n{subtitle}', fontsize=11, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def print_summary(datasets: List[List[Dict[str, Any]]], labels: List[str]):
    """Print a summary table of the benchmark results for multiple datasets."""
    for data, label in zip(datasets, labels):
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
    # Single prefix
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks QWen3"
    
    # Multiple prefixes for comparison
    python visualize.py --prefix fw_pyroworks trtllm_pyroworks --label "Fireworks" "TRT-LLM"
    
    # With percentiles
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks" --percentile p50 p95 p99
    
    # Custom metric
    python visualize.py --prefix accounts_pyroworks --label "Pyroworks" --metric output_sequence_length
        """
    )
    
    parser.add_argument('--prefix', type=str, nargs='+', required=True,
                        help='Prefix(es) to match artifact directories (e.g., "accounts_pyroworks" or multiple for comparison)')
    parser.add_argument('--label', type=str, nargs='+', required=True,
                        help='Label(s) for the plot legend/title (must match number of prefixes)')
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
    
    # Validate prefix and label counts match
    if len(args.prefix) != len(args.label):
        print(f"Error: Number of prefixes ({len(args.prefix)}) must match number of labels ({len(args.label)})")
        sys.exit(1)
    
    # Find and load JSON files for each prefix
    datasets = []
    labels = args.label
    
    for prefix, label in zip(args.prefix, args.label):
        print(f"\nSearching for benchmarks with prefix: {prefix}")
        json_files = find_json_files(args.artifacts_dir, prefix)
        
        if not json_files:
            print(f"Error: No benchmark files found with prefix '{prefix}' in {args.artifacts_dir}")
            sys.exit(1)
        
        print(f"Found {len(json_files)} benchmark file(s) for '{label}':")
        for f in json_files:
            print(f"  - {f}")
        
        # Load benchmark data
        data = load_benchmark_data(json_files)
        datasets.append(data)
    
    # Print summary if requested
    if args.summary:
        print_summary(datasets, labels)
    
    # Generate plots
    show_plot = not args.no_show
    
    if args.plot_type == 'latency':
        plot_metric(datasets, labels, args.metric, args.percentile, args.output, show_plot)
    elif args.plot_type == 'throughput':
        output = args.output.replace('.png', '_throughput.png') if args.output else None
        plot_throughput(datasets, labels, output, show_plot)
    elif args.plot_type == 'combined':
        plot_combined(datasets, labels, args.percentile, args.output, show_plot)
    elif args.plot_type == 'all':
        base_output = args.output.replace('.png', '') if args.output else None
        plot_metric(datasets, labels, args.metric, args.percentile, 
                   f"{base_output}_latency.png" if base_output else None, show_plot)
        plot_throughput(datasets, labels,
                       f"{base_output}_throughput.png" if base_output else None, show_plot)
        plot_combined(datasets, labels, args.percentile,
                     f"{base_output}_combined.png" if base_output else None, show_plot)


if __name__ == '__main__':
    main()
