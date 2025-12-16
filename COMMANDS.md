```bash
docker exec -it aidan bash
cd ~/benchmark/llm_bench

conda create --prefix ./env python=3.10

source ~/miniconda3/bin/activate
conda activate ./env
pip install uv
uv pip install -r requirements.txt



```