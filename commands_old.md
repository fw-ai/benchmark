


# TODO:
mkdir -p ~/work/shared/text-models/hf/phi3-vision-128k-instruct/
rsync -avz --progress aidan@192.222.54.149:/shared/text-models/hf/phi3-vision-128k-instruct/ ~/work/shared/text-models/hf/phi3-vision-128k-instruct/
kubectl exec -it aidan-vlm-ubuntu-test-pod -- mkdir -p /fireworks/models/hf/phi3-vision-128k-instruct/
kubectl cp ~/work/shared/text-models/hf/phi3-vision-128k-instruct/ aidan-vlm-ubuntu-test-pod:/home/aidan/models/hf/phi3-vision-128k-instruct/
kubectl exec -it aidan-vlm-ubuntu-test-pod -- /bin/bash
fireworks-llm-server /home/aidan/models/hf/phi3-vision-128k-instruct/phi3-vision-128k-instruct/ --port 8000
kubectl port-forward pod/aidan-vlm-ubuntu-test-pod 8000:8000


du -sh /home
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok
ngrok add auth-token 
ngrok http --url=lasting-swan-large.ngrok-free.app 8000

kubectl port-forward pod/aidan-vlm-ubuntu-test-pod 8000:8000

vllm serve \
    microsoft/Phi-3-vision-128k-instruct \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --quantization fp8

kubectl config get-contexts
kubectl config use-context "US_TEXAS_2 (lambda us-south-2)"
# https://jarvis.fireworks-ai.cloud/clusters
kubectl config use-context "US_IOWA_1 (gcp us-central1-a)"

firectl -a teo-0b3810 scale mq8yiuy4 --replica-count 1

kubectl get nodes -o wide
kubectl describe nodes | grep -i gpu
kubectl get pods -A -o wide | grep -i gpu
kubectl get nodes | grep -e NAME -e "192-222-5"

kubectl apply -f pod.yaml
kubectl apply -f pod2.yaml
kubectl apply -f pod-gcp.yaml

kubectl get pods

kubectl describe pod aidan-vlm-ubuntu-test-pod

kubectl logs aidan-vlm-ubuntu-test-pod -c download-models -f
kubectl logs aidan-vlm-ubuntu-test-pod -f

kubectl delete pod aidan-vlm-ubuntu-test-pod

kubectl exec -it aidan-vlm-ubuntu-test-pod -- /bin/bash



# p90 latencies:

For a single GPU, we're actually worse than vLLM:
So assuming they're smart enough to replicate one vLLM server for each GPU
(and able to work through all the edge cases and issues with that)

Us:
- 7 QPS: 1700ms
- 8 QPS: 2200ms
- 9 QPS: 3400ms
- 10 QPS: 11000ms
Them:
- 7 QPS: 680ms
- 8 QPS: 840ms
- 9 QPS: 1200ms
- 10 QPS: 3000ms



# Text completion

p90 latencies:
Us:
- 7 QPS: 590ms
- 8 QPS: 670ms
- 9 QPS: 1100ms
- 10 QPS: 1300ms
vLLM:
- 7 QPS: 560ms
- 8 QPS: 660ms
- 9 QPS: 670ms
- 10 QPS: 670ms

p99 latencies:
Us:
- 7 QPS: 920ms
- 8 QPS: 870ms
- 9 QPS: 1400ms
- 10 QPS: 1600ms
vLLM:
- 7 QPS: 790ms
- 8 QPS: 680ms
- 9 QPS: 1100ms
- 10 QPS: 700ms


```
raise RuntimeError(                                                                                                                                    RuntimeError: Worker failed with error 'Error in memory profiling. Initial free memory 49586110464, current free memory 75894423552. This happens when the GPU memory was not properly cleaned up before initializing the vLLM instance.', please check the stack trace above for the root cause                      Traceback (most recent call last):    
```


```bash
cd ~/home/fireworks/serving/dev && make docker-build
cd ~/home/fireworks/serving/dev && make docker-run
cd ~/home/fireworks/serving/dev && make docker-attach
cd ~/benchmark/llm_bench && source ~/miniconda3/bin/activate && conda activate ./env
docker exec -it --workdir /home/aidan/fireworks -- aidan /bin/bash

source ~/miniconda3/bin/activate

conda create --prefix ./env python=3.10

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

conda activate ./env
pip install uv
uv pip install -r requirements.txt

locust \
    -t 5min \
    -u 100 \
    -r 100 \
    --prompt-text @aledade_prod.jsonl \
    -o 80 \
    --chat \
    --no-stream \
    --logprobs 1 \
    --temperature 0.0 \
    --qps 1 \
    --summary-file vllm-1qps.csv \
    --provider vllm \
    -H http://localhost:8000 \
    --model microsoft/Phi-3-vision-128k-instruct \
    --headless

for qps in {7..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-text @aledade_prod.jsonl \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/vllm-fp8-$qps-qps.csv \
        --provider vllm \
        -H http://localhost:8000 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done

for qps in {7..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-text @/home/aidan/fireworks/aledade_prod.jsonl \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/fireworks-fp8-$qps-qps.csv \
        --provider fireworks \
        -H http://localhost:8000 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done


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
        --summary-file benchmark_results/fireworks-text-$qps-qps.csv \
        --provider fireworks \
        -H http://localhost:80 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done

```


```bash
# vLLM
cd ~/home/fireworks/serving/dev && make docker-attach
cd ~/benchmark/llm_bench && source ~/miniconda3/bin/activate && conda activate ./vllm


conda create --prefix ./vllm python=3.10
conda activate ./vllm
pip install uv
uv pip install vllm

CUDA_VISIBLE_DEVICES=6 vllm serve microsoft/Phi-3-vision-128k-instruct \
  --trust-remote-code \
  --no-enable-prefix-caching
# All 8 GPUS
vllm serve \
    microsoft/Phi-3-vision-128k-instruct \
    --trust-remote-code \
    --tensor-parallel-size 8
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve microsoft/Phi-3-vision-128k-instruct --trust-remote-code --tensor-parallel-size 4
CUDA_VISIBLE_DEVICES=6,7 vllm serve microsoft/Phi-3-vision-128k-instruct --trust-remote-code --tensor-parallel-size 2
CUDA_VISIBLE_DEVICES=6 \
    vllm serve \
    microsoft/Phi-3-vision-128k-instruct \
    --trust-remote-code \
    --no-enable-prefix-caching
CUDA_VISIBLE_DEVICES=6 \
    vllm serve \
    microsoft/Phi-3-vision-128k-instruct \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --quantization fp8

# text_completion container
# releases/text-completion/4.0.49
CUDA_VISIBLE_DEVICES=6 \
    fireworks-llm-server \
    /shared/text-models/hf/phi3-vision-128k-instruct/ \
    --uninitialized-weights-precision float8 \
    --convert-to-precision float8 \
    --port 8000
```


apt-get update
apt-get install wget
apt-get install git

git config --global user.email "aidando73@gmail.com"
git config --global user.name "Aidan Do"

apt-get install gh
gh auth login

git clone https://github.com/aidando73/benchmark-private.git
cd benchmark-private/llm_bench
conda create --prefix ./env python=3.12 -y
conda activate ./env
pip install uv
uv pip install -r requirements.txt




mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda create --prefix ./vllm python=3.12 -y
conda activate ./vllm
pip install uv
uv pip install vllm







for qps in {9..9}; do
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
        --prompt-randomize \
        --summary-file benchmark_results/vllm-text-fp8-$qps-qps.csv \
        --provider vllm \
        -H http://localhost:8000 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done


for qps in {7..10}; do
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
        --summary-file benchmark_results/fireworks-text-no-prompt-caching-$qps-qps.csv \
        --provider fireworks \
        -H http://localhost:80 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done

mkdir -p ~/work/shared/text-models/hf/phi3-vision-128k-instruct/
rsync -avz --progress aidan@192.222.54.149:/shared/text-models/hf/phi3-vision-128k-instruct/ ~/work/shared/text-models/hf/phi3-vision-128k-instruct/


firectl -a pyroworks list deployments
firectl -a pyroworks create deployment "accounts/aledade/models/behavextractphi3vecw20250222" --accelerator-type NVIDIA_H100_80GB --min-replica-count 1
watch -c "firectl -a pyroworks list deployments"

firectl -a teo-0b3810 get deployment mq8yiuy4
firectl -a pyroworks delete deployment cl2g8mxj
firectl -a teo-0b3810 scale mq8yiuy4 --replica-count 1
firectl -a teo-0b3810 scale mq8yiuy4 --replica-count 0

kubectl describe pod teo-0b3810-mq8yiuy4-5bd799b595-sz9c7
kubectl exec -it teo-0b3810-mq8yiuy4-5bd799b595-sz9c7 -- /bin/bash

for qps in {7..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-text @/home/aidan/fireworks/aledade_prod.jsonl \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/fireworks-direct-$qps-qps.csv \
        --provider fireworks \
        -H https://teo-0b3810-mq8yiuy4.direct.fireworks.ai \
        --model "accounts/aledade/models/behavextractphi3vecw20250222#accounts/teo-0b3810/deployments/mq8yiuy4" \
        --headless \
        --api-key __api_key__
        # --prompt-tokens 2300 \
done



for qps in {7..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-text @/home/aidan/fireworks/aledade_prod.jsonl \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/fireworks-fp8-extra-metrics-$qps-qps.csv \
        --provider fireworks \
        -H http://localhost:8000 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done

for qps in {7..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-text @aledade_prod.jsonl \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/vllm-fp8-$qps-qps.csv \
        --provider vllm \
        -H http://localhost:80 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done

for qps in {7..10}; do
    locust \
        -t 3min \
        -u 100 \
        -r 100 \
        --prompt-text @aledade_prod.jsonl \
        -o 80 \
        --chat \
        --no-stream \
        --logprobs 1 \
        --temperature 0.0 \
        --qps $qps \
        --summary-file benchmark_results/fireworks-fp8-$qps-qps.csv \
        --provider fireworks \
        -H http://localhost:80 \
        --model microsoft/Phi-3-vision-128k-instruct \
        --headless
done
```