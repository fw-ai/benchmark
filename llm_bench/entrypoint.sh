#!/bin/bash

set -e;

echo "VLLM_ENDPOINT=$VLLM_ENDPOINT"
echo "ADAPTIVE_ENDPOINT=$ADAPTIVE_ENDPOINT"
echo "ADAPTIVE_API_KEY=$ADAPTIVE_API_KEY"

HARMONY_ENDPOINT=http://adaptive-harmony-0.adaptive-harmony-hdls-svc.default.svc.cluster.local:50053
ADAPTIVE_VERSION=$(curl -s $HARMONY_ENDPOINT/version_info | jq -r 'try .image_tag // empty' || echo "")
VLLM_REF="2025-03-05-1720"

export GPU_NAME=""

OUTPUT_DIR=/llm-benchmarks
PREFIX=$(date +"%Y-%m-%d-%H%M")

echo $PREFIX > $OUTPUT_DIR/latest.txt

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

echo "Generate plots"
python plotting.py --model Llama-3.1-8b --output-tokens 128 --input-files $OUTPUT_DIR/nocache/$PREFIX-adaptive.csv $OUTPUT_DIR/nocache/$VLLM_REF-vllm.csv --output-file $OUTPUT_DIR/reports/$PREFIX-vllm-adaptive-nocache.html --extra-header "Adaptive $ADAPTIVE_VERSION vs. vllm 0.7.1 (randomized prompts)"
python plotting.py --model Llama-3.1-8b --output-tokens 128 --input-files $OUTPUT_DIR/perfectcache/$PREFIX-adaptive.csv $OUTPUT_DIR/perfectcache/$VLLM_REF-vllm.csv --output-file $OUTPUT_DIR/reports/$PREFIX-vllm-adaptive-perfectcache.html --extra-header "Adaptive $ADAPTIVE_VERSION vs. vllm 0.7.1 (randomized prompts)"