#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET_FILE="$SCRIPT_DIR/preset.json"

# Default values
MODEL=""
URL=""
ENDPOINT_TYPE=""
STREAMING=""
CONCURRENCY_LIST=()
REQUEST_COUNT=""
TOKENIZER=""
SEQUENCE_DISTRIBUTION=""
EXTRA_INPUTS=""
API_KEY="${FIREWORKS_API_KEY:-}"
NO_GPU_TELEMETRY="true"
NO_SERVER_METRICS="true"
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
            # Load concurrency as array (supports both array and single value)
            CONCURRENCY_JSON=$(jq -r ".[\"$PRESET\"].concurrency" "$PRESET_FILE")
            if [[ $(echo "$CONCURRENCY_JSON" | jq -r 'type') == "array" ]]; then
                CONCURRENCY_LIST=()
                while IFS= read -r val; do
                    CONCURRENCY_LIST+=("$val")
                done < <(echo "$CONCURRENCY_JSON" | jq -r '.[]')
            else
                CONCURRENCY_LIST=("$CONCURRENCY_JSON")
            fi
            REQUEST_COUNT=$(jq -r ".[\"$PRESET\"][\"request-count\"] // empty" "$PRESET_FILE")
            TOKENIZER=$(jq -r ".[\"$PRESET\"].tokenizer // empty" "$PRESET_FILE")
            SEQUENCE_DISTRIBUTION=$(jq -r ".[\"$PRESET\"][\"sequence-distribution\"] // empty" "$PRESET_FILE")
            EXTRA_INPUTS=$(jq -r ".[\"$PRESET\"][\"extra-inputs\"] // empty" "$PRESET_FILE")
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
        --request-count)
            REQUEST_COUNT="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --sequence-distribution)
            SEQUENCE_DISTRIBUTION="$2"
            shift 2
            ;;
        --extra-inputs)
            EXTRA_INPUTS="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --no-gpu-telemetry)
            NO_GPU_TELEMETRY="true"
            shift
            ;;
        --no-server-metrics)
            NO_SERVER_METRICS="true"
            shift
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

# Default to concurrency of 1 if not specified
if [[ ${#CONCURRENCY_LIST[@]} -eq 0 ]]; then
    CONCURRENCY_LIST=(1)
fi

# Iterate over concurrency values
for CONCURRENCY in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with concurrency: $CONCURRENCY"
    echo "=========================================="

    # Build command
    CMD=(aiperf profile --model "$MODEL" --url "$URL")

    if [[ -n "$ENDPOINT_TYPE" ]]; then
        CMD+=(--endpoint-type "$ENDPOINT_TYPE")
    fi

    if [[ "$STREAMING" == "true" ]]; then
        CMD+=(--streaming)
    fi

    CMD+=(--concurrency "$CONCURRENCY")

    if [[ -n "$REQUEST_COUNT" ]]; then
        CMD+=(--request-count "$REQUEST_COUNT")
    fi

    if [[ -n "$TOKENIZER" ]]; then
        CMD+=(--tokenizer "$TOKENIZER")
    fi

    if [[ -n "$SEQUENCE_DISTRIBUTION" ]]; then
        CMD+=(--sequence-distribution "$SEQUENCE_DISTRIBUTION")
    fi

    if [[ -n "$EXTRA_INPUTS" ]]; then
        CMD+=(--extra-inputs "$EXTRA_INPUTS")
    fi

    if [[ -n "$API_KEY" ]]; then
        CMD+=(--api-key "$API_KEY")
    fi

    if [[ "$NO_GPU_TELEMETRY" == "true" ]]; then
        CMD+=(--no-gpu-telemetry)
    fi

    if [[ "$NO_SERVER_METRICS" == "true" ]]; then
        CMD+=(--no-server-metrics)
    fi

    # Add extra arguments
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    # Run command
    echo "Running command: ${CMD[@]}"
    "${CMD[@]}"
done
