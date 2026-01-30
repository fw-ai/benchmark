#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET_FILE="$SCRIPT_DIR/preset.json"

# Default values
MODEL=""
URL=""
ENDPOINT_TYPE=""
STREAMING=""
CONCURRENCY=""
REQUEST_COUNT=""
TOKENIZER=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            if [[ ! -f "$PRESET_FILE" ]]; then
                echo "Error: Preset file not found: $PRESET_FILE"
                exit 1
            fi
            # Check if preset exists
            if ! jq -e ".[\"$PRESET\"]" "$PRESET_FILE" > /dev/null 2>&1; then
                echo "Error: Unknown preset '$PRESET'"
                echo "Available presets: $(jq -r 'keys | join(", ")' "$PRESET_FILE")"
                exit 1
            fi
            # Load preset values
            MODEL=$(jq -r ".[\"$PRESET\"].model // empty" "$PRESET_FILE")
            URL=$(jq -r ".[\"$PRESET\"].url // empty" "$PRESET_FILE")
            ENDPOINT_TYPE=$(jq -r ".[\"$PRESET\"][\"endpoint-type\"] // empty" "$PRESET_FILE")
            STREAMING=$(jq -r ".[\"$PRESET\"].streaming // empty" "$PRESET_FILE")
            CONCURRENCY=$(jq -r ".[\"$PRESET\"].concurrency // empty" "$PRESET_FILE")
            REQUEST_COUNT=$(jq -r ".[\"$PRESET\"][\"request-count\"] // empty" "$PRESET_FILE")
            TOKENIZER=$(jq -r ".[\"$PRESET\"].tokenizer // empty" "$PRESET_FILE")
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --url)
            URL="$2"
            shift 2
            ;;
        --endpoint-type)
            ENDPOINT_TYPE="$2"
            shift 2
            ;;
        --streaming)
            STREAMING="true"
            shift
            ;;
        --no-streaming)
            STREAMING="false"
            shift
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --request-count)
            REQUEST_COUNT="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required parameters
if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required (or use --preset)"
    exit 1
fi

if [[ -z "$URL" ]]; then
    echo "Error: --url is required (or use --preset)"
    exit 1
fi

# Build command
CMD=(aiperf profile --model "$MODEL" --url "$URL")

if [[ -n "$ENDPOINT_TYPE" ]]; then
    CMD+=(--endpoint-type "$ENDPOINT_TYPE")
fi

if [[ "$STREAMING" == "true" ]]; then
    CMD+=(--streaming)
fi

if [[ -n "$CONCURRENCY" ]]; then
    CMD+=(--concurrency "$CONCURRENCY")
fi

if [[ -n "$REQUEST_COUNT" ]]; then
    CMD+=(--request-count "$REQUEST_COUNT")
fi

if [[ -n "$TOKENIZER" ]]; then
    CMD+=(--tokenizer "$TOKENIZER")
fi

# Add extra arguments
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

# Run command
echo "Running command: ${CMD[@]}"
"${CMD[@]}"
