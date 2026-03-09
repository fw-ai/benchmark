# Benchmark / Load-testing Suite by Fireworks.ai

## LLM benchmarking

The load test is designed to simulate continuous production load and minimize effect of model generation behavior:
* variation in generation parameters
* continuous request stream with varying distribution and load levels
* force generation of exact number of output tokens (for most providers)
* specified load test duration

Supported providers and API flavors:
* OpenAI API compatible endpoints:
  * [Fireworks.ai](https://app.fireworks.ai) public or private deployments
  * VLLM
  * Anyscale Endpoints
  * OpenAI
* Text Generation Inference (TGI) / HuggingFace Endpoints
* Together.ai
* NVidia Triton server:
  * Legacy HTTP endpoints (no streaming)
  * LLM-focused endpoints (with or without streaming)

Supported API types:
* Chat completions (`/v1/chat/completions`)
* Text completions (`/v1/completions`)
* Embeddings (`/v1/embeddings`)
* Rerank (`/v1/rerank`)

Captured metrics:
* Overall latency
* Number of generated tokens
* Sustained requests throughput (QPS)
* Time to first token (TTFT) for streaming
* Per token latency for streaming

Metrics summary can be exported to CSV. This way multiple configuration can be scripted over. CSV file can be imported to Google Sheets/Excel or Jupyter for further analysis.

See [`llm_bench`](llm_bench) folder for detailed usage.

See [`llm_bench/benchmark_suite.ipynb`](llm_bench/benchmark_suite.ipynb) for a detailed example of how to use the load test script and run different types of benchmark suites.
