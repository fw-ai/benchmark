#!/bin/bash


provider="vllm"
summary_file="vllm.csv"
model="meta-llama/Llama-3.1-8B-Instruct"
api_key=""

while getopts ":p:s:u:m:k:" opt; do
  case $opt in
    p) provider="$OPTARG"
    ;;
    s) summary_file="$OPTARG"
    ;;
    u) url="$OPTARG"
    ;;
    m) model="$OPTARG"
    ;;
    k) api_key="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


lengths=(128 256 512 1024 2048 4096)
# lengths=(256 512 1024 2048 4096)
# lengths=(256 2048)
qps=(0.125 0.5 1 2 4 68 10 12 14 16 18 20)
# qps=(8)

for length in "${lengths[@]}"; do
    for q in "${qps[@]}"; do
        echo "Running load test with $length input token size and $q qps"
        echo ""
        locust \
            -H $url \
            -m $model \
            --tokenizer meta-llama/Llama-3.1-8B-Instruct \
            --provider $provider \
            --qps $q \
            -u 500 \
            -r 500 \
            -p $length \
            -o 128 \
            --chat \
            --stream \
            --summary-file $summary_file \
            -t 60 \
            -k $api_key

        sleep 5
    done
done

