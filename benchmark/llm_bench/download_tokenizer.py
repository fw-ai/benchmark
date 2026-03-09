#!/usr/bin/env python3
"""
Download a Hugging Face tokenizer (and config) to a local directory.
Use the output path with --tokenizer to avoid loading the hub under Locust/gevent.

Example:
  python download_tokenizer.py cross-encoder/ms-marco-MiniLM-L6-v2
  # Saves to ./tokenizers/cross-encoder_ms-marco-MiniLM-L6-v2/

  python download_tokenizer.py cross-encoder/ms-marco-MiniLM-L6-v2 /path/to/save
  # Saves to /path/to/save
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face tokenizer to a local directory."
    )
    parser.add_argument(
        "model_id",
        help="Hugging Face model id (e.g. cross-encoder/ms-marco-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "save_dir",
        nargs="?",
        default=None,
        help="Directory to save to (default: ./tokenizers/<model_id with / replaced by _>)",
    )
    args = parser.parse_args()

    # Import here so this script can run without loading under gevent
    from transformers import AutoTokenizer

    model_id = args.model_id.strip()
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        safe_name = model_id.replace("/", "_")
        save_dir = os.path.join(base, "tokenizers", safe_name)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading tokenizer: {model_id}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved to: {save_dir}", file=sys.stderr)
    print(save_dir)


if __name__ == "__main__":
    main()
