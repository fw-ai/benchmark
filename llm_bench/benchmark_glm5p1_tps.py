#!/usr/bin/env python3
"""
GLM 5.1 TPS (Tokens Per Second) Benchmark Suite

This script runs a series of benchmarks to measure and optimize TPS for GLM 5.1
deployments. It helps identify the optimal configuration for maximum throughput.

Usage:
    python benchmark_glm5p1_tps.py --host <deployment_url> --api-key <key> [options]

Example:
    python benchmark_glm5p1_tps.py \
        --host https://api.fireworks.ai/inference \
        --api-key $FIREWORKS_API_KEY \
        --model accounts/fireworks/models/glm-5p1

The script will:
1. Run baseline TPS measurement
2. Test different concurrency levels to find saturation point
3. Test different batch sizes and token lengths
4. Generate a summary report with recommendations
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    concurrency: int
    prompt_tokens: int
    max_tokens: int
    qps: float
    avg_ttft_ms: float
    avg_latency_per_token_ms: float
    avg_total_latency_ms: float
    completion_tokens_avg: float
    num_requests: int
    p50_total_latency: float
    p99_total_latency: float
    
    @property
    def tps(self) -> float:
        """Calculate tokens per second (output TPS)."""
        if self.avg_latency_per_token_ms > 0:
            return 1000.0 / self.avg_latency_per_token_ms * self.concurrency
        return 0.0
    
    @property
    def throughput_tps(self) -> float:
        """Calculate aggregate throughput TPS based on QPS and tokens."""
        return self.qps * self.completion_tokens_avg


def run_benchmark(
    host: str,
    api_key: str,
    model: str,
    concurrency: int,
    prompt_tokens: int,
    max_tokens: int,
    duration: str = "1min",
    tokenizer: Optional[str] = None,
    extra_args: Optional[list] = None,
) -> Optional[BenchmarkResult]:
    """Run a single benchmark and return results."""
    
    results_dir = f"results/glm5p1_tps_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    cmd = [
        "locust",
        "-f", os.path.join(os.path.dirname(__file__), "load_test.py"),
        "--headless",
        "--only-summary",
        "-H", host,
        "--provider", "fireworks",
        "-m", model,
        "-k", api_key,
        "-u", str(concurrency),
        "-r", str(min(concurrency, 10)),
        "-t", duration,
        "-p", str(prompt_tokens),
        "-o", str(max_tokens),
        "--max-tokens-distribution", "constant",
        "--chat",
        "--stream",
        "--temperature", "1.0",
        "--csv", f"{results_dir}/stats",
        "--summary-file", f"{results_dir}/summary.csv",
    ]
    
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: concurrency={concurrency}, prompt={prompt_tokens}, max_tokens={max_tokens}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode != 0:
            print(f"Benchmark failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        print(result.stdout)
        
        summary_file = f"{results_dir}/summary.csv"
        if not os.path.exists(summary_file):
            print(f"Summary file not found: {summary_file}")
            return None
        
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            
            return BenchmarkResult(
                concurrency=concurrency,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                qps=float(row.get('Qps', 0)),
                avg_ttft_ms=float(row.get('Time To First Token', 0) or 0),
                avg_latency_per_token_ms=float(row.get('Latency Per Token', 0) or 0),
                avg_total_latency_ms=float(row.get('Total Latency', 0)),
                completion_tokens_avg=float(row.get('Completion Tokens', max_tokens)),
                num_requests=int(row.get('Num Requests', 0)),
                p50_total_latency=float(row.get('P50 Total Latency', 0) or 0),
                p99_total_latency=float(row.get('P99 Total Latency', 0) or 0),
            )
            
    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def run_concurrency_sweep(
    host: str,
    api_key: str,
    model: str,
    concurrency_levels: list[int],
    prompt_tokens: int = 512,
    max_tokens: int = 256,
    duration: str = "1min",
    tokenizer: Optional[str] = None,
) -> list[BenchmarkResult]:
    """Run benchmarks at different concurrency levels to find saturation point."""
    
    results = []
    for concurrency in concurrency_levels:
        result = run_benchmark(
            host=host,
            api_key=api_key,
            model=model,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            duration=duration,
            tokenizer=tokenizer,
        )
        if result:
            results.append(result)
            print(f"\nResult: TPS={result.throughput_tps:.1f}, QPS={result.qps:.2f}, "
                  f"Latency/token={result.avg_latency_per_token_ms:.2f}ms")
        
        time.sleep(5)
    
    return results


def run_token_length_sweep(
    host: str,
    api_key: str,
    model: str,
    concurrency: int,
    token_configs: list[tuple[int, int]],
    duration: str = "1min",
    tokenizer: Optional[str] = None,
) -> list[BenchmarkResult]:
    """Run benchmarks with different prompt/output token configurations."""
    
    results = []
    for prompt_tokens, max_tokens in token_configs:
        result = run_benchmark(
            host=host,
            api_key=api_key,
            model=model,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            duration=duration,
            tokenizer=tokenizer,
        )
        if result:
            results.append(result)
            print(f"\nResult: TPS={result.throughput_tps:.1f}, QPS={result.qps:.2f}, "
                  f"Latency/token={result.avg_latency_per_token_ms:.2f}ms")
        
        time.sleep(5)
    
    return results


def generate_report(results: list[BenchmarkResult], output_file: str):
    """Generate a summary report from benchmark results."""
    
    if not results:
        print("No results to report")
        return
    
    report_lines = [
        "=" * 80,
        "GLM 5.1 TPS Benchmark Report",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "=" * 80,
        "",
        "RESULTS SUMMARY",
        "-" * 80,
        f"{'Concurrency':>12} {'Prompt':>8} {'MaxTok':>8} {'QPS':>8} {'TPS':>10} {'TTFT(ms)':>10} {'Lat/Tok':>10} {'P99(ms)':>10}",
        "-" * 80,
    ]
    
    best_tps = max(results, key=lambda r: r.throughput_tps)
    
    for r in results:
        marker = " *" if r == best_tps else ""
        report_lines.append(
            f"{r.concurrency:>12} {r.prompt_tokens:>8} {r.max_tokens:>8} "
            f"{r.qps:>8.2f} {r.throughput_tps:>10.1f} {r.avg_ttft_ms:>10.1f} "
            f"{r.avg_latency_per_token_ms:>10.2f} {r.p99_total_latency:>10.1f}{marker}"
        )
    
    report_lines.extend([
        "-" * 80,
        "",
        "ANALYSIS",
        "-" * 80,
        f"Best TPS achieved: {best_tps.throughput_tps:.1f} tokens/second",
        f"  - Concurrency: {best_tps.concurrency}",
        f"  - Prompt tokens: {best_tps.prompt_tokens}",
        f"  - Max tokens: {best_tps.max_tokens}",
        f"  - QPS: {best_tps.qps:.2f}",
        f"  - Avg latency per token: {best_tps.avg_latency_per_token_ms:.2f}ms",
        "",
        "RECOMMENDATIONS",
        "-" * 80,
    ])
    
    if best_tps.throughput_tps < 150:
        report_lines.extend([
            "- TPS is below expected B200 performance (~300 TPS for this model size)",
            "- Consider checking:",
            "  1. GPU utilization on the deployment",
            "  2. Speculative decoding configuration",
            "  3. Batch size limits on the server",
            "  4. Network latency between client and server",
        ])
    elif best_tps.throughput_tps < 250:
        report_lines.extend([
            "- TPS is moderate but below optimal B200 performance",
            "- Consider increasing concurrency or checking server-side batching",
        ])
    else:
        report_lines.extend([
            "- TPS is within expected range for B200 deployment",
        ])
    
    if best_tps.avg_ttft_ms > 500:
        report_lines.append("- High TTFT suggests prefill bottleneck - consider prompt caching")
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    report = "\n".join(report_lines)
    print(report)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    csv_file = output_file.replace('.txt', '.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'concurrency', 'prompt_tokens', 'max_tokens', 'qps', 'tps',
            'ttft_ms', 'latency_per_token_ms', 'total_latency_ms',
            'p50_latency_ms', 'p99_latency_ms', 'num_requests'
        ])
        for r in results:
            writer.writerow([
                r.concurrency, r.prompt_tokens, r.max_tokens, r.qps, r.throughput_tps,
                r.avg_ttft_ms, r.avg_latency_per_token_ms, r.avg_total_latency_ms,
                r.p50_total_latency, r.p99_total_latency, r.num_requests
            ])
    
    print(f"\nReport saved to: {output_file}")
    print(f"CSV data saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="GLM 5.1 TPS Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--host", "-H",
        required=True,
        help="Deployment URL (e.g., https://api.fireworks.ai/inference)",
    )
    parser.add_argument(
        "--api-key", "-k",
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--model", "-m",
        default="accounts/fireworks/models/glm-5p1",
        help="Model identifier (default: accounts/fireworks/models/glm-5p1)",
    )
    parser.add_argument(
        "--tokenizer",
        help="Path to HuggingFace tokenizer for accurate token counting",
    )
    parser.add_argument(
        "--duration", "-t",
        default="1min",
        help="Duration for each benchmark run (default: 1min)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer configurations",
    )
    parser.add_argument(
        "--concurrency-only",
        action="store_true",
        help="Only run concurrency sweep",
    )
    parser.add_argument(
        "--output-dir",
        default="results/glm5p1_tps_suite",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    print("\n" + "=" * 80)
    print("GLM 5.1 TPS Benchmark Suite")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Model: {args.model}")
    print(f"Duration per test: {args.duration}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    print("\n\n### PHASE 1: Concurrency Sweep ###")
    print("Finding optimal concurrency level for maximum TPS...")
    
    if args.quick:
        concurrency_levels = [1, 4, 8, 16, 32]
    else:
        concurrency_levels = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    
    concurrency_results = run_concurrency_sweep(
        host=args.host,
        api_key=args.api_key,
        model=args.model,
        concurrency_levels=concurrency_levels,
        prompt_tokens=512,
        max_tokens=256,
        duration=args.duration,
        tokenizer=args.tokenizer,
    )
    all_results.extend(concurrency_results)
    
    if concurrency_results:
        best_concurrency = max(concurrency_results, key=lambda r: r.throughput_tps).concurrency
        print(f"\nBest concurrency found: {best_concurrency}")
    else:
        best_concurrency = 16
        print(f"\nUsing default concurrency: {best_concurrency}")
    
    if not args.concurrency_only:
        print("\n\n### PHASE 2: Token Length Sweep ###")
        print("Testing different prompt/output token configurations...")
        
        if args.quick:
            token_configs = [
                (256, 128),
                (512, 256),
                (1024, 512),
            ]
        else:
            token_configs = [
                (128, 64),
                (256, 128),
                (512, 256),
                (512, 512),
                (1024, 256),
                (1024, 512),
                (2048, 256),
            ]
        
        token_results = run_token_length_sweep(
            host=args.host,
            api_key=args.api_key,
            model=args.model,
            concurrency=best_concurrency,
            token_configs=token_configs,
            duration=args.duration,
            tokenizer=args.tokenizer,
        )
        all_results.extend(token_results)
    
    report_file = f"{output_dir}/tps_benchmark_report.txt"
    generate_report(all_results, report_file)
    
    print("\n" + "=" * 80)
    print("Benchmark suite complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
