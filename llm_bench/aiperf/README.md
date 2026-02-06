# Benchmark with aiperf

```bash
uv venv --python=3.10 

uv pip install -r aiperf/requirements.txt

./aiperf/run_aiperf.sh --preset h200-local

./aiperf/run_aiperf.sh --preset b200-local
```