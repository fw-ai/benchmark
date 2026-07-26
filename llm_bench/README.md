# LLM Load test

Please refer to the [`benchmark_suite.ipynb`](benchmark_suite.ipynb) for a detailed example of how to use the load_test.py script and run different types of benchmark suites.

## Installation

The load test relies on [Locust package](https://locust.io/). Install it from pip.

```bash
pip install -r requirements.txt
```

Then run the commands described below from the enclosing directory. Locust will pick up the settings from `locust.conf` automatically.

## Usage

The load test script exercises LLM generation endpoint under varying load. See below for the common configuration options. Check `--help` for the full list.

### Target

- `-H`: target endpoint URL (preceding `/v1/...`). E.g. `-H http://localhost` or `-H https://api.fireworks.ai/inference`. Defaults to `localhost:80`.
- (optional) `-m`: model to send requests too. Can be omitted for a local test if the server has a single model loaded only.
- (optional) `--provider`: provider name like `fireworks` or `openai`. APIs have slight differences that the script accounts for. If omitted the script tries to guess based on URI and API return information. Must be specified for non-OpenAI-compatible providers like Triton.
- `-k`: API key to be passed as `Authorization: Bearer ...`.

### Rate of requests

There are several primary modes the script can be used:

1. **Fixed concurrency**. N workers are created. Each sends a request, waits for the response and then sends the next request. Thus as concurrency increases, the server will get more loaded and latency will grow. Usually increasing concurrency beyond some point doesn't increase throughput and just leads to growing latency.
   - `-u`: the number of concurrent workers to spawn (standard Locust argument)
   - `-r`: the rate per second of spawning concurrent workers. If processing workload takes a while (more than several seconds), it makes sense to set this value to something lower than `-u` for a gradual ramp-up to avoid request bursts.
   - (optionally) `--burst <period in seconds>`: synchronizes all N workers to issue requests in one go with the specified interval. The maximum latency should be less than the period, otherwise some workers may fall behind.

2. **Fixed QPS**. The script ensures that input requests are issued at specific times to average out at the specified rate per second. If the target QPS is too high and the server is overloaded it will likely drop additional requests or stall.
   - `--qps`: the desired rate of requests per second. Can be a fractional number, e.g. `0.1`.
   - `-u <high number> -r <high number>`: needs to be set to a sufficiently high value to allow generating the target QPS. The script will complain if it's too low. Passing something like `-u 100 -r 100` is a good choice.
   - (optional) `--qps-distribution`: specify how to space out requests. Default is `constant` meaning evenly spaced out. `exponential` is an option simulating [Poisson distribution](https://en.wikipedia.org/wiki/Traffic_generation_model#Poisson_traffic_model).

### Workload

Input is read from --dataset, which is either:
- `limerics`: default dataset. Requires --tokenizer to be passed. Will be used to auto-generate realistic prompts.
- `@`-prefixed, specifies a path to JSONL file, used to read contents of each request.

The number of tokens to generate is sampled on every request from a given distribution:
- `-o`/`--max-tokens`: maximum number of tokens to generate. If --max-tokens-distribution is non-constant this is going to be the mean of the distribution.
- `--max-tokens-distribution`: specifies probability distribution to use.
- `--max-tokens-range` Specifies "the width" of the distribution (e.g. stddev for "normal" distribution). Specified value `alpha` is relative to `max-tokens`. Default is 0.3 so most of the range falls in "3 sigma" region.
- `--max-tokens-cap`: specify upper bound to "truncate" the probability distribution. The lower bound is always 1 token. This allows to sample from "truncated normal" or "truncated exponential" distributions.

Based on the above settings the following distributions are supported:
- `constant`: use `--max_tokens` value on every request
- `uniform`: sample from the range `[max_tokens - max_tokens * alpha, max_tokens + max_tokens * alpha]`
- `normal`: sample from gaussian distribution `N(max_tokens, max_tokens * alpha)`
- `exponential`: sample from exponential distribution with the mean `max_tokens`. `alpha` is ignored

The benchmark makes the best effort to ensure the desired `max_tokens` number is respected:
- for providers that support it, it passes `ignore_eos` or `min_tokens` parameter to avoid early stopping
- the default prompt is a lengthy code generation request that usually doesn't stop early
- it verifies the number of tokens actually generated and prints warnings on mismatch. Different providers use varying mechanisms of returning generated number of tokens. For some of them `--logprobs` might be needed in the streaming mode.
- optionally, `--tokenizer` can be passed specifying Huggingface tokenizer to be used to count the output tokens on client side.

Generation options:
- `--chat`: specify to call chat API instead of raw completions
- `--stream`: stream the result back. Enabling this gives "time to first token" and "time per token" metrics
- (optional) `--logprobs`: corresponds to `logprobs` API parameter. For some providers, it's needed for output token counting in streaming mode.

Session mode (progressive cache / agentic-traffic simulation):
- `--session-mode`: simulate conversation history like Claude Code / Cursor / Codex traffic. Each locust user (connection) runs a progressively growing multi-turn conversation: it starts with a single small user message and, after each generation, appends a fresh user turn while reusing the prior messages as an exact-continuation cacheable prefix. This is the pattern that produces prompt-cache hits on serving stacks which require exact message continuation (e.g. DeepSeek-V4), unlike the default fixed-prefix-replay workload which does not. Requires `--chat`.
- `--session-turn-tokens`: with `--session-mode`, the number of *new* prompt tokens appended each turn (the uncached delta). Defaults to `--prompt-tokens` minus `--prompt-cache-max-len` (e.g. `-p 60000 -pcml 54000` -> 6000 new tokens per turn).
- The conversation resets once it reaches `2 * --prompt-tokens`, so the *average* requested prompt length across the run stays around `--prompt-tokens` even though each session grows from ~`session-turn-tokens` up to `2 * --prompt-tokens`.
- Each turn's user content is drawn from the existing `limericks`/`code` datasets and ends with the usual translate-to-Spanish/C++ suffix so responses stay coherent and the assistant turn becomes part of the cached prefix on the next turn.
- Recommended companion flags to avoid request clumping: a meaningful spawn rate (`-r 4` or so) and `--max-tokens-distribution uniform`.

Embeddings and rerank options:
- `--embeddings`: use the `/v1/embeddings` API instead of completions
- `--rerank`: use the `/v1/rerank` API. The generated prompt text is split into documents (by paragraph), and `--rerank-query` is used as the query.
- `--rerank-query`: the search query string for rerank requests. Defaults to a generic query if not specified.
- `--rerank-top-n`: number of top results to return from the rerank endpoint.
- `--rerank-return-documents` / `--no-rerank-return-documents`: whether to include document text in the rerank response (default: true).

### Writing results

Locust prints out the detailed summary including quantiles of various metrics. Additionally, the script prints out the summary block at the very end of the output that includes the model being tested.

When comparing multiple configurations, it's useful to aggregate results together:

- `--summary-file`: Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, it writes out the header first.
- `-t`: duration (e.g. `5min`) for which to run the test (standard Locust option). It's particularly useful when scripting multiple runs. By default, the test runs without a limit until Ctrl+C is pressed.

The typical workflow would be to run benchmark several times appending to the same CSV file. The resulting file can be imported into a spreadsheet or pandas for further analysis.

### Custom prompts

Sometimes it's necessary to replay exact prompts, for example in the case of embedding images.
`--dataset` option can be used in this case to specify a file with .jsonl extension (starting with an ampersand, e.g. `@prompt.jsonl`.).
jsonl files will be read line-by-line. Each line has to have a valid JSON object, which will be used to form the resulting API request.
Examples:

Chat dataset (--chat option):
```
{"messages": [{"role": "user", "content": "Write a poem about a cat"}], "temperature": 0.9}
{"messages": [{"role": "user", "content": "Write a poem about a dog"}], "temperature": 1}
```

Non-chat dataset (--no-chat option):
```
{"prompt": "One two three four"}
{"prompt": "Five six seven eight"}
```


## Examples

Download tokenizer for the model being benchmarked from Huggingface.
```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /models/Meta-Llama-3-8B-Instruct   --include '*.json'
export TOKENIZER=/models/Meta-Llama-3-8B-Instruct
```

Maintain fixed 8 requests concurrency against local deployment:

```bash
locust -u 8 -r 2 -p 512 -o 128
```

Call streaming chat API locally with the request issued every 2 seconds. Run for 1 minute and save results to `results.csv`:

```bash
locust -t 1min -u 100 -r 100 -p 512 -o 128 --stream --chat --qps 0.5 --summary-file results.csv
```

Benchmark Fireworks public deployment deployment with 1 request only:

```bash
locust -u 1 -H https://api.fireworks.ai/inference -p 128 -o 200 --api-key $FIREWORKS_API_KEY --model=accounts/fireworks/models/llama-v3p1-8b-instruct
```

Benchmark Fireworks public deployment with 1 request and 2 images (1024w x 1024h and 3084w x 1080h):

```bash
locust -u 1  -H https://api.fireworks.ai/inference -p 128 -o 200 --api-key $FIREWORKS_API_KEY --model=accounts/fireworks/models/llama-v3p1-8b-instruct --chat --prompt-images-with-resolutions 1024x1024 3084x1080
```

Benchmark Fireworks rerank deployment with a single request and 115k prompt tokens:

```bash
locust -u 1 -r 2 -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY -m "accounts/fireworks/models/qwen3-reranker-8b" --rerank --prompt-tokens 115000 -t 3min --tokenizer /path/to/tokenizer --summary-file rerank_results.csv
```

Benchmark Fireworks rerank with a custom query and top-5 results:

```bash
locust -u 1 -r 2 -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY -m "accounts/fireworks/models/qwen3-reranker-8b" --rerank --rerank-query "How do I reset my password?" --rerank-top-n 5 --prompt-tokens 4096 -t 1min --tokenizer /path/to/tokenizer
```

Benchmark OpenAI deployment reading prompts from a file at 1 QPS:

```bash
locust --dataset '@input.jsonl' -u 1 -H https://api.openai.com -o 200 --api-key $OPENAI_API_KEY --model=gpt-3.5-turbo --chat
```

Simulate agentic / Claude Code style progressive-cache traffic (avg prompt ~60K, 54K cached, 32 concurrent users, ~600-token turns, requests de-clumped via uniform output-length distribution and a 4/s spawn rate). Each user grows a multi-turn conversation from ~6K up to ~120K, then resets:

```bash
locust -f llm_bench/load_test.py -t 10m -u 32 -r 4 \
  -p 60000 -pcml 54000 -o 600 --max-tokens-distribution uniform \
  --session-mode --chat --stream \
  -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY \
  -m accounts/fireworks/models/<model> --tokenizer $TOKENIZER \
  --summary-file session_results.csv
```

## UI mode

Instead of relying on textual data, it's also possible to plot the results in Grafana.

```bash
pip install locust locust-plugins
locust-compose up
```

This starts your local Postgre and Grafana. Grafana is available at http://127.0.0.1:3000 (sometimes logs don't print out).

Then run the test as specified above with an additional argument:

```bash
locust --config locust-grafana.conf ...
```

This starts the load test locally and pushes results into Grafana in real-time. Besides the actual requests, we push additional metrics (e.g. time per token) as separate fake requests to get stats aggregation. Make sure to remove them from aggregation when viewing the graphs.

Other settings for Locust are in `./locust.conf`. You may start Locust in non-headless mode, but its UI is very basic and misses advanced stats aggregation capabilities.
