#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET_FILE="$SCRIPT_DIR/preset.json"
LOAD_TEST_FILE="$SCRIPT_DIR/../load_test.py"

# Default values
ENGINE="aiperf"  # aiperf (default) or locust
MODEL=""
URL=""
ENDPOINT_TYPE=""
PROVIDER=""  # For locust: fireworks, vllm, etc.
STREAMING=""
CONCURRENCY_LIST=()
REQUEST_COUNT=""
TOKENIZER=""
EXTRA_INPUTS=""
API_KEY="${FIREWORKS_API_KEY:-}"
NO_GPU_TELEMETRY="true"
NO_SERVER_METRICS="true"
EXTRA_ARGS=()

# Token configuration
OUTPUT_TOKENS=""
PROMPT_TOKENS=""
PROMPT_STD_MULT="3.3"  # Multiplier for prompt std calculation
OUTPUT_STD_MULT="3.3"  # Multiplier for output std calculation
DATASET=""  # For aiperf: sharegpt, etc. (uses --public-dataset instead of --sequence-distribution)

# Locust-specific defaults
OUTPUT_DIR=""
CHAT="true"  # Default to chat mode

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --engine)
            ENGINE="$2"
            if [[ "$ENGINE" != "aiperf" && "$ENGINE" != "locust" ]]; then
                echo "Error: --engine must be 'aiperf' or 'locust'"
                exit 1
            fi
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --output-tokens|-o)
            OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --prompt-tokens|-p)
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        --prompt-std)
            PROMPT_STD_MULT="$2"
            shift 2
            ;;
        --output-std)
            OUTPUT_STD_MULT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chat)
            CHAT="true"
            shift
            ;;
        --no-chat)
            CHAT="false"
            shift
            ;;
        --concurrency|-u)
            # Parse comma-separated or space-separated concurrency values
            IFS=',' read -ra CONC_VALS <<< "$2"
            CONCURRENCY_LIST+=("${CONC_VALS[@]}")
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run LLM benchmarks using either aiperf or locust engine."
            echo ""
            echo "Engine Selection:"
            echo "  --engine ENGINE         Benchmark engine: 'aiperf' (default) or 'locust'"
            echo ""
            echo "Common Options:"
            echo "  --preset NAME           Load configuration from preset.json"
            echo "  --model MODEL           Model name/path (required)"
            echo "  --url URL               API endpoint URL (required)"
            echo "  --api-key KEY           API key (default: \$FIREWORKS_API_KEY)"
            echo "  --concurrency, -u N     Concurrency level(s), comma-separated for multiple runs"
            echo "  --request-count N       Number of requests to run (shared between engines)"
            echo "  --streaming             Enable streaming mode (default: from preset)"
            echo "  --tokenizer PATH        HuggingFace tokenizer path"
            echo ""
            echo "aiperf-specific Options:"
            echo "  --endpoint-type TYPE    API type: openai-chat, openai-completions, etc."
            echo "  --dataset NAME          Public dataset (e.g., sharegpt) - uses --public-dataset instead of --sequence-distribution"
            echo "  --extra-inputs JSON     Extra inputs for requests"
            echo "  --no-gpu-telemetry      Disable GPU telemetry"
            echo "  --no-server-metrics     Disable server metrics"
            echo ""
            echo "locust-specific Options:"
            echo "  --provider PROVIDER     API provider: fireworks, vllm, sglang, openai"
            echo "  --output-tokens, -o N   Output tokens per request"
            echo "  --prompt-tokens, -p N   Prompt token count"
            echo "  --prompt-std MULT       Prompt std divisor for aiperf (default: 3.3)"
            echo "  --output-std MULT       Output std divisor for aiperf (default: 3.3)"
            echo "  --chat                  Use /v1/chat/completions API (default)"
            echo "  --no-chat               Use /v1/completions API"
            echo "  --output-dir DIR        Output directory for results"
            echo ""
            echo "Examples:"
            echo "  # Using aiperf (default)"
            echo "  $0 --preset fw-pyroworks --request-count 100"
            echo ""
            echo "  # Using locust with Fireworks"
            echo "  $0 --engine locust --provider fireworks \\"
            echo "      --model accounts/fireworks/models/llama-v3p2-3b-instruct \\"
            echo "      --url https://api.fireworks.ai/inference \\"
            echo "      --concurrency 8 --request-count 100"
            echo ""
            echo "  # Using locust with vLLM"
            echo "  $0 --engine locust --provider vllm \\"
            echo "      --model meta-llama/Llama-3.2-3B-Instruct \\"
            echo "      --url http://localhost:8000 \\"
            echo "      --concurrency 1,4,8,16 --request-count 100"
            exit 0
            ;;
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
            PROVIDER=$(jq -r ".[\"$PRESET\"].provider // empty" "$PRESET_FILE")
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
            EXTRA_INPUTS=$(jq -r ".[\"$PRESET\"][\"extra-inputs\"] // empty" "$PRESET_FILE")
            # Token configuration preset values
            OUTPUT_TOKENS=$(jq -r ".[\"$PRESET\"][\"output-tokens\"] // empty" "$PRESET_FILE")
            PROMPT_TOKENS=$(jq -r ".[\"$PRESET\"][\"prompt-tokens\"] // empty" "$PRESET_FILE")
            PRESET_PROMPT_STD=$(jq -r ".[\"$PRESET\"][\"prompt-std\"] // empty" "$PRESET_FILE")
            PRESET_OUTPUT_STD=$(jq -r ".[\"$PRESET\"][\"output-std\"] // empty" "$PRESET_FILE")
            PRESET_DATASET=$(jq -r ".[\"$PRESET\"].dataset // empty" "$PRESET_FILE")
            [[ -n "$PRESET_PROMPT_STD" ]] && PROMPT_STD_MULT="$PRESET_PROMPT_STD"
            [[ -n "$PRESET_OUTPUT_STD" ]] && OUTPUT_STD_MULT="$PRESET_OUTPUT_STD"
            [[ -n "$PRESET_DATASET" ]] && DATASET="$PRESET_DATASET"
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
        --request-count)
            REQUEST_COUNT="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
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

# Infer provider from URL if not specified (for locust engine)
if [[ -z "$PROVIDER" ]]; then
    if [[ "$URL" == *"fireworks.ai"* ]]; then
        PROVIDER="fireworks"
    elif [[ -n "$ENDPOINT_TYPE" ]]; then
        # Map endpoint-type to provider for locust compatibility
        if [[ "$ENDPOINT_TYPE" == "openai-chat" || "$ENDPOINT_TYPE" == "openai-completions" ]]; then
            PROVIDER="vllm"  # Default to vllm for generic OpenAI-compatible endpoints
        fi
    fi
fi

# Infer endpoint-type from provider if not specified (for aiperf engine)
if [[ -z "$ENDPOINT_TYPE" && -n "$PROVIDER" ]]; then
    if [[ "$PROVIDER" == "fireworks" || "$PROVIDER" == "vllm" || "$PROVIDER" == "sglang" || "$PROVIDER" == "openai" ]]; then
        if [[ "$CHAT" == "true" ]]; then
            ENDPOINT_TYPE="openai-chat"
        else
            ENDPOINT_TYPE="openai-completions"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Engine: $ENGINE"
echo "Model: $MODEL"
echo "URL: $URL"
if [[ "$ENGINE" == "aiperf" ]]; then
    echo "Endpoint Type: $ENDPOINT_TYPE"
else
    echo "Provider: $PROVIDER"
fi
echo "=========================================="

# Iterate over concurrency values
for CONCURRENCY in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with concurrency: $CONCURRENCY"
    echo "=========================================="

    if [[ "$ENGINE" == "aiperf" ]]; then
        ###############################
        # AIPERF ENGINE
        ###############################
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

        # Use public-dataset if specified, otherwise generate sequence-distribution
        if [[ -n "$DATASET" ]]; then
            CMD+=(--public-dataset "$DATASET")
        elif [[ -n "$PROMPT_TOKENS" && -n "$OUTPUT_TOKENS" ]]; then
            # Generate sequence-distribution from prompt-tokens and output-tokens
            # Format: {prompt_tokens}|{prompt_std},{output_tokens}|{output_std}
            # std = tokens * std_mult / 100
            PROMPT_STD=$(awk "BEGIN {print int($PROMPT_TOKENS * $PROMPT_STD_MULT / 100)}")
            OUTPUT_STD=$(awk "BEGIN {print int($OUTPUT_TOKENS * $OUTPUT_STD_MULT / 100)}")
            CMD+=(--sequence-distribution "${PROMPT_TOKENS}|${PROMPT_STD},${OUTPUT_TOKENS}|${OUTPUT_STD}:100")
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
    else
        ###############################
        # LOCUST ENGINE
        ###############################
        
        # Validate locust-specific requirements
        if [[ ! -f "$LOAD_TEST_FILE" ]]; then
            echo "Error: load_test.py not found at: $LOAD_TEST_FILE"
            exit 1
        fi

        if [[ -z "$PROVIDER" ]]; then
            echo "Error: --provider is required for locust engine (fireworks, vllm, sglang, openai)"
            exit 1
        fi

        # Create output directory
        if [[ -z "$OUTPUT_DIR" ]]; then
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
            OUTPUT_DIR="$SCRIPT_DIR/../results/${SAFE_MODEL}_concurrency${CONCURRENCY}_${TIMESTAMP}"
        fi
        mkdir -p "$OUTPUT_DIR"

        CMD=(locust
            --headless
            --only-summary
            -H "$URL"
            --provider "$PROVIDER"
            --model "$MODEL"
            -f "$LOAD_TEST_FILE"
            --html "$OUTPUT_DIR/report.html"
            --csv "$OUTPUT_DIR/stats"
        )

        # API key
        if [[ -n "$API_KEY" ]]; then
            CMD+=(--api-key "$API_KEY")
        fi

        # Chat mode
        if [[ "$CHAT" == "true" ]]; then
            CMD+=(--chat)
        else
            CMD+=(--no-chat)
        fi

        # Streaming mode
        if [[ "$STREAMING" == "true" ]]; then
            CMD+=(--stream)
        elif [[ "$STREAMING" == "false" ]]; then
            CMD+=(--no-stream)
        fi

        # Concurrency mode: users = concurrency, spawn rate = concurrency
        CMD+=(-u "$CONCURRENCY" -r "$CONCURRENCY")

        # Request count (shared with aiperf)
        if [[ -n "$REQUEST_COUNT" ]]; then
            CMD+=(--max-requests "$REQUEST_COUNT")
        else
            # Default to 100 requests if not specified
            CMD+=(--max-requests 100)
        fi

        # Output tokens
        if [[ -n "$OUTPUT_TOKENS" ]]; then
            CMD+=(--max-tokens "$OUTPUT_TOKENS")
        fi

        # Prompt tokens (input tokens)
        if [[ -n "$PROMPT_TOKENS" ]]; then
            CMD+=(--prompt-tokens "$PROMPT_TOKENS")
        fi

        # Tokenizer
        if [[ -n "$TOKENIZER" ]]; then
            CMD+=(--tokenizer "$TOKENIZER")
        fi

        # Add extra arguments
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            CMD+=("${EXTRA_ARGS[@]}")
        fi

        # Reset OUTPUT_DIR for next iteration
        OUTPUT_DIR=""
    fi

    # Run command
    echo "Running command: ${CMD[@]}"
    "${CMD[@]}"
done
