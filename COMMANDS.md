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


for qps in 2 4 6 8 10 12 14; do
    model="gemini-2.5-flash-preview-04-17"
    summary_file="results/$model-$(date +%Y-%m-%d_%H-%M)/$qps-qps.csv"
    mkdir -p $(dirname $summary_file)
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-tokens 1267 \
        --prompt-images-with-resolutions 570x570 \
        -o 601 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file $summary_file \
        --provider gemini \
        -H https://generativelanguage.googleapis.com \
        --model $model \
        --headless
done

```


- 3 million Etsy listing per day
- Complete within ~20 hrs