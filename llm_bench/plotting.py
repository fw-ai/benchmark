import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def main(args):
    # Read the CSV data
    dfs = []
    for idx, f in enumerate(args.input_files):
        this_df = pd.read_csv(f)
        if args.provider_suffixes:
            suffix = args.provider_suffixes[idx]
            this_df["Provider"] = this_df["Provider"].astype(str) + f"-{suffix}"
        dfs.append(this_df)

    df = pd.concat(dfs, axis=0)

    # Create the HTML file
    html_output = []
    html_output.append(
        """
    <html>
    <head>
        <title>Benchmark Results Analysis</title>
        <style>
            .plot-container {
                width: 100%;
                margin: 20px 0;
            }
            h1 {
                text-align: center;
                margin: 40px 0 20px 0;
            }
            h2 {
                text-align: left;
                margin: 40px 0 20px 0;
            }
        </style>
    </head>
    <body>
    """
    )

    gpu_name = os.environ.get("GPU_NAME", "")
    html_output.append(
        f"<h1>Model:{args.model}   |   Output tokens:{args.output_tokens}   |   Time per test (s):60   |   GPU:1 x {gpu_name} </h1>"
    )
    if args.extra_header:
        html_output.append(f"<h1>{args.extra_header}</h1>")
    # html_output.append("")

    # Get unique prompt token values
    prompt_tokens = sorted(df["Prompt Tokens"].unique())

    line_and_fill_colors = [
        ("blue", "135, 206, 250, 0.4"),
        ("red", "255, 182, 193, 0.4"),
        ("orange", "255, 200, 124, 0.4"),
        ("pink", "255, 192, 203, 0.4"),
        ("green", "144, 238, 144, 0.4"),
    ]
    # Create plots for each prompt token value
    for token_value in prompt_tokens:
        # Filter data for current prompt token value
        token_df = df[df["Prompt Tokens"] == token_value]

        # Create figure with two subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "% incomplete requests/total requests vs. Concurrency",
                "P90 Time to First Token vs. Concurrency",
                "P90 Time per Output Token vs. Concurrency",
                "P90 Total Latency vs. Concurrency",
            ),
            vertical_spacing=0.1,
        )

        # Plot data for each provider
        # for provider in ["vllm", "adaptive"]:
        for idx, provider in enumerate(df["Provider"].unique()):
            provider_df = token_df[token_df["Provider"] == provider]

            # Incomplete requests/total requests
            ratio = round((provider_df["Incomplete Requests"] / provider_df["Total Requests"]) * 100, 2)
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=ratio,
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # P90 Time to First Token
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Time To First Token"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # P90 Latency per Token
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Latency Per Token"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

            # P90 Total Latency
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Total Latency"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

        # Update layout
        fig.update_layout(
            # title_text=f"Input Tokens: {int(token_value)}",
            height=1000,
            width=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        # Update axes
        fig.update_xaxes(title_text="Concurrency (QPS)", row=1, col=1)
        fig.update_xaxes(title_text="Concurrency (QPS)", row=2, col=1)
        fig.update_xaxes(title_text="Concurrency (QPS)", row=3, col=1)
        fig.update_xaxes(title_text="Concurrency (QPS)", row=4, col=1)
        fig.update_yaxes(title_text="% incomplete requests/total requests", row=1, col=1)
        fig.update_yaxes(title_text="TTFT (ms)", row=2, col=1)
        fig.update_yaxes(title_text="TPOT(ms)", row=3, col=1)
        fig.update_yaxes(title_text="Total latency(ms)", row=4, col=1)

        # Add to HTML output
        html_output.append(f"<h2>Input Tokens: {int(token_value)}</h2>")
        html_output.append('<div class="plot-container">')
        html_output.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        html_output.append("</div>")

    # Close HTML file
    html_output.append("</body></html>")

    # Write to file
    print("Writing HTML")
    with open(args.output_file, "w") as f:
        f.write("\n".join(html_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--model", type=str, required=True, help="Name of the benchmarked model")
    parser.add_argument("--output-tokens", type=int, required=True, help="Number of output tokens")
    parser.add_argument(
        "--input-files",  # Name of the argument
        nargs="+",
        type=str,
        help="Provide 2 results files to plot",
        required=True,
    )
    parser.add_argument(
        "--provider-suffixes",  # Name of the argument
        nargs="+",
        type=str,
        help="Provide suffixes to attach to provider names, so you can compare same provider with different configs",
        required=False,
    )
    parser.add_argument(
        "--output-file", type=str, required=False, default="results.html", help="Number of output tokens"
    )
    parser.add_argument("--extra-header", type=str, required=False, help="Add an h1 header to top of page")

    args = parser.parse_args()
    assert len(args.input_files) <= 5, "Can only compare 5 result files at a time"
    if args.provider_suffixes:
        assert len(args.input_files) == len(
            args.provider_suffixes
        ), "If passing suffixes, you must pass one per input file"

    main(args)
