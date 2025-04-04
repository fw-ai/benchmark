#!/bin/bash

set -e;

export GPU_NAME=""
OUTPUT_DIR=/llm-benchmarks
PREFIX=$(date +"%Y-%m-%d-%H%M")

BENCHMARKS_PAGE=https://inference-benchmarks.tech-adaptive-ml.com/reports
HARMONY_ENDPOINT=http://adaptive-harmony-0.adaptive-harmony-hdls-svc.default.svc.cluster.local:50053
ADAPTIVE_VERSION=$(curl -s $HARMONY_ENDPOINT/version_info | jq -r 'try .image_tag // empty' || echo "")
VLLM_REF="2025-03-05-1720"
ADAPTIVE_REF=$(cat $OUTPUT_DIR/latest.txt)
ADAPTIVE_REF_LATEST_VERSION=$(cat $OUTPUT_DIR/adaptive-latest-version.txt)

echo "VLLM_ENDPOINT=$VLLM_ENDPOINT"
echo "VLLM_REF=$VLLM_REF"
echo "ADAPTIVE_ENDPOINT=$ADAPTIVE_ENDPOINT"
echo "ADAPTIVE_API_KEY=$ADAPTIVE_API_KEY"
echo "ADAPTIVE_VERSION=$ADAPTIVE_VERSION"
echo "ADAPTIVE_REF=$ADAPTIVE_REF"
echo "ADAPTIVE_REF_LATEST_VERSION=$ADAPTIVE_REF_LATEST_VERSION"

mkdir -p $OUTPUT_DIR/nocache
mkdir -p $OUTPUT_DIR/perfectcache
mkdir -p $OUTPUT_DIR/reports

echo "adaptive - 100% cache hit rate"
./launch_all.sh -s $OUTPUT_DIR/perfectcache/$PREFIX-adaptive.csv -u $ADAPTIVE_ENDPOINT -p adaptive -m test -k $ADAPTIVE_API_KEY

echo "adaptive - 0% cache hit rate (randomized beginning tokens for each prompt)"
./launch_all.sh -s $OUTPUT_DIR/nocache/$PREFIX-adaptive.csv -u $ADAPTIVE_ENDPOINT -p adaptive -m test -k $ADAPTIVE_API_KEY -r

#echo "vllm - 100% cache hit rate"
#./launch_all.sh -s $OUTPUT_DIR/perfectcache/$PREFIX-vllm.csv -u $VLLM_ENDPOINT

#echo "vllm - 0% cache hit rate (randomized beginning tokens for each prompt)"
#./launch_all.sh -s $OUTPUT_DIR/nocache/$PREFIX-vllm.csv -u $VLLM_ENDPOINT -r

# echo "Generate plots (adaptive vs vllm)"
# python plotting.py --model Llama-3.1-8b --output-tokens 128 --input-files $OUTPUT_DIR/nocache/$PREFIX-adaptive.csv $OUTPUT_DIR/nocache/$VLLM_REF-vllm.csv --output-file $OUTPUT_DIR/reports/$PREFIX-vllm-adaptive-nocache.html --extra-header "Adaptive $ADAPTIVE_VERSION vs. vllm 0.7.1 (randomized prompts)"
# python plotting.py --model Llama-3.1-8b --output-tokens 128 --input-files $OUTPUT_DIR/perfectcache/$PREFIX-adaptive.csv $OUTPUT_DIR/perfectcache/$VLLM_REF-vllm.csv --output-file $OUTPUT_DIR/reports/$PREFIX-vllm-adaptive-perfectcache.html --extra-header "Adaptive $ADAPTIVE_VERSION vs. vllm 0.7.1 (randomized prompts)"

echo "Generate plots (current adaptive [$ADAPTIVE_VERSION] vs previous adaptive [$ADAPTIVE_REF_LATEST_VERSION] vs vllm)"
python plotting.py --model Llama-3.1-8b --output-tokens 128 --input-files $OUTPUT_DIR/nocache/$PREFIX-adaptive.csv $OUTPUT_DIR/nocache/$ADAPTIVE_REF-adaptive.csv $OUTPUT_DIR/nocache/$VLLM_REF-vllm.csv --output-file $OUTPUT_DIR/reports/$PREFIX-adaptive-$ADAPTIVE_VERSION-vs-$ADAPTIVE_REF_LATEST_VERSION-nocache.html --extra-header "Adaptive $ADAPTIVE_VERSION vs. $ADAPTIVE_REF_LATEST_VERSION vs vllm (randomized prompts)" --provider-suffixes $ADAPTIVE_VERSION $ADAPTIVE_REF_LATEST_VERSION "0.7"
python plotting.py --model Llama-3.1-8b --output-tokens 128 --input-files $OUTPUT_DIR/perfectcache/$PREFIX-adaptive.csv $OUTPUT_DIR/perfectcache/$ADAPTIVE_REF-adaptive.csv $OUTPUT_DIR/perfectcache/$VLLM_REF-vllm.csv --output-file $OUTPUT_DIR/reports/$PREFIX-adaptive-$ADAPTIVE_VERSION-vs-$ADAPTIVE_REF_LATEST_VERSION-perfectcache.html --extra-header "Adaptive $ADAPTIVE_VERSION vs. $ADAPTIVE_REF_LATEST_VERSION vs vllm (randomized prompts)" --provider-suffixes $ADAPTIVE_VERSION $ADAPTIVE_REF_LATEST_VERSION "0.7"

echo "mark Reference version for next benchmark runs"
echo $PREFIX > $OUTPUT_DIR/latest.txt
echo $ADAPTIVE_VERSION > $OUTPUT_DIR/adaptive-latest-version.txt

PERFECT_CACHE_REPORT_URL=https://inference-benchmarks.tech-adaptive-ml.com/reports/$PREFIX-adaptive-$ADAPTIVE_VERSION-vs-$ADAPTIVE_REF_LATEST_VERSION-perfectcache.html
NO_CACHE_REPORT_URL=https://inference-benchmarks.tech-adaptive-ml.com/reports/$PREFIX-adaptive-$ADAPTIVE_VERSION-vs-$ADAPTIVE_REF_LATEST_VERSION-nocache.html

echo "publish slack message"
curl -X POST -H 'Content-type: application/json' --data "{
  \"text\": \"*Latest Inference Benchmark Reports Available: Adaptive [$ADAPTIVE_VERSION] vs [$ADAPTIVE_REF_LATEST_VERSION]*\n
  🔗 <${PERFECT_CACHE_REPORT_URL}|Perfect Cache>\n
  🔗 <${NO_CACHE_REPORT_URL}|No Cache>\"
}" "$SLACK_WEBHOOK_URL"

