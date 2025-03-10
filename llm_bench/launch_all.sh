#!/bin/bash


provider="vllm"
summary_file="vllm.csv"
model="meta-llama/Llama-3.1-8B-Instruct"
api_key="not-relevant-for-vllm"
randomize=false

while getopts "p:s:u:m:k:r" opt; do
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
    r) randomize=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


lengths=(8192 16384)
qps=(0.125 0.5 1 2 4 6 8 10 12 14 16 18 20)

for length in "${lengths[@]}"; do
    for q in "${qps[@]}"; do
        echo "Running load test with $length input token size and $q qps"
        echo ""
        locust_command="locust \
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
            -t 300 \
            -k $api_key"
          
        if [ "$randomize" = true ]; then
            locust_command+=" --prompt-randomize"
        fi

        eval $locust_command

        sleep 5
    done
done

