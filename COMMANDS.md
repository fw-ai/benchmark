```bash
cd ~/work/benchmark-fork/llm_bench

conda create --prefix ./env python=3.10 -y
conda activate ./env
pip install uv
uv pip install -r requirements.txt

locust -t 1min -u 100 -r 100 -p 512 -o 128 --stream --chat --qps 0.5 --summary-file results.csv
```