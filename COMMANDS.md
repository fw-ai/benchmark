```bash
cd ~/work/benchmark-fork/llm_bench

conda create --prefix ./env python=3.10 -y
conda activate ./env
pip install uv
uv pip install -r requirements.txt

locust -t 1min \
    -u 1 \
    -r 1 \
    -p 512 \
    -o 128 \
    --stream \
    --chat \
    --qps 0.5 \
    --provider=fireworks \
    --model=accounts/fireworks/models/llama-v3p1-8b-instruct \
    --api-key $FIREWORKS_API_KEY \
    --prompt-images-with-resolutions 1024x1024 2048x3803 \
    -H https://api.fireworks.ai/inference \
    --summary-file results.csv

cd ~/home/fireworks/serving/dev && make docker-attach
locust -t 1min \
    -u 1 \
    -r 1 \
    -p 512 \
    -o 128 \
    --stream \
    --qps 0.5 \
    --provider=fireworks \
    --model=test \
    --prompt-images-with-resolutions 1024x1024 2048x3803 \
    -H http://localhost:80 \
    --summary-file results.csv
    # --chat
```

--image-resolutions
--with-images-by-resolutions
--images-by-resolutions
--images-by-urls
--images-with-resolutions
--images-with-urls