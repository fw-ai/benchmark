"""
Extract latency stats from Locust batch_*_stats.csv files in Results/
and write a single CSV in the same format as the original (Concurrency -> Num Documents,
LPC -> Total Latency, plus TTFT). Reads from llm_bench/Results/ by default.
"""

import os
import pandas as pd
import re


def extract_batch_size(filename: str) -> int | None:
    """Extract batch size from filename like batch_100_stats.csv -> 100."""
    match = re.search(r"batch_(\d+)_stats\.csv", filename)
    return int(match.group(1)) if match else None


def process_stats(base_dir: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Read all batch_*_stats.csv in base_dir and produce one CSV with
    Num Documents, Requests/s, Total Latency stats, TTFT stats (same shape as original).
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    results = []
    for filename in sorted(os.listdir(base_dir)):
        batch_size = extract_batch_size(filename)
        if batch_size is None:
            continue
        stats_file = os.path.join(base_dir, filename)
        if not os.path.isfile(stats_file):
            continue

        df = pd.read_csv(stats_file)

        # Prefer METRIC rows (total_latency, time_to_first_token); fall back to POST
        total_row = None
        ttft_row = None

        if "Type" in df.columns and "Name" in df.columns:
            metric_df = df[df["Type"] == "METRIC"]
            if not metric_df.empty:
                total_df = metric_df[metric_df["Name"] == "total_latency"]
                ttft_df = metric_df[metric_df["Name"] == "time_to_first_token"]
                if not total_df.empty:
                    total_row = total_df.iloc[0]
                if not ttft_df.empty:
                    ttft_row = ttft_df.iloc[0]

        if total_row is None or ttft_row is None:
            post_df = df[(df["Type"] == "POST")] if "Type" in df.columns else pd.DataFrame()
            if not post_df.empty:
                post_row = post_df.iloc[0]
                if total_row is None:
                    total_row = post_row
                if ttft_row is None:
                    ttft_row = post_row

        if total_row is None or ttft_row is None:
            print(f"Warning: Could not find total_latency/TTFT metrics in {filename}, skipping.")
            continue

        # Same column naming as original script (LPC -> Total Latency for rerank)
        result = {
            "Num Documents": batch_size,
            "Requests/s": total_row["Requests/s"],
            "Total Latency Median (ms)": total_row["Median Response Time"],
            "Total Latency Average (ms)": total_row["Average Response Time"],
            "Total Latency p50 (ms)": total_row["50%"],
            "Total Latency p90 (ms)": total_row["90%"],
            "Total Latency p95 (ms)": total_row["95%"],
            "Total Latency p99 (ms)": total_row["99%"],
            "Total Latency p99.9 (ms)": total_row["99.9%"],
            "TTFT Median (ms)": ttft_row["Median Response Time"],
            "TTFT Average (ms)": ttft_row["Average Response Time"],
            "TTFT p50 (ms)": ttft_row["50%"],
            "TTFT p90 (ms)": ttft_row["90%"],
            "TTFT p95 (ms)": ttft_row["95%"],
            "TTFT p99 (ms)": ttft_row["99%"],
            "TTFT p99.9 (ms)": ttft_row["99.9%"],
        }
        results.append(result)

    if not results:
        print("No batch_*_stats.csv files found or no valid metrics.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("Num Documents")

    if output_path is None:
        output_path = os.path.join(base_dir, "latency_stats_output.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(results_df)} rows.")
    return results_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    process_stats(results_dir)


if __name__ == "__main__":
    main()
