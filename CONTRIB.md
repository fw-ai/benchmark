# Contributing


## Components

- `vllm_helm`: vllm helm chart install.
- Benchmark job docker: `llm_bench/Dockerfile`. build and push image: `./llm_bench/build.sh` (requires aws access to adaptive account)
- `llm_bench/job.yaml`: benchmark job k8s manifest which runs benchmark docker. program
- `llm_bench/ui.yaml`: k8s install for an website served by nginx which serves as an index of all benchamrks results and history.