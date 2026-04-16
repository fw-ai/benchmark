#!/usr/bin/env python3
"""
Download a HuggingFace tokenizer locally given a model's HF id.

Usage:
    python download_tokenizer.py <hf_model_id> [--output-dir DIR]

Examples:
    python download_tokenizer.py Qwen/Qwen3-Reranker-8B
    python download_tokenizer.py Qwen/Qwen3-Reranker-8B --output-dir ./tokenizers/qwen3-reranker-8b
    python download_tokenizer.py meta-llama/Llama-3.1-8B-Instruct --hf-token $HF_TOKEN
"""

import argparse
import os
import sys


def download_tokenizer(model_id: str, output_dir: str, hf_token: str = None):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers is not installed. Run: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading tokenizer for: {model_id}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    kwargs = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    tokenizer.save_pretrained(output_dir)

    print(f"Tokenizer saved to: {output_dir}")
    print(f"Pass to load tests with: --tokenizer {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace tokenizer locally given a model's HF id."
    )
    parser.add_argument(
        "model_id",
        help="HuggingFace model id (e.g. Qwen/Qwen3-Reranker-8B)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to save the tokenizer. Defaults to ./tokenizers/<model-name>",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token for gated models. Falls back to $HF_TOKEN env var.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join("tokenizers", args.model_id.split("/")[-1])
    download_tokenizer(args.model_id, output_dir, args.hf_token)


if __name__ == "__main__":
    main()
