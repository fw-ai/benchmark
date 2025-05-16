```bash
cd ~/home/fireworks/serving/dev && make docker-build
cd ~/home/fireworks/serving/dev && make docker-run
cd ~/home/fireworks/serving/dev && make docker-attach
cd ~/benchmark/llm_bench && source ~/miniconda3/bin/activate && conda activate ./env
docker exec -it aidan /bin/bash

conda create --prefix ./env python=3.10


source ~/miniconda3/bin/activate ./env
pip install uv
uv pip install -r requirements.txt


for qps in {5..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-tokens 2300 \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/vllm-text-$qps-qps.csv \
        --provider vllm \
        -H http://localhost:8000 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done

```