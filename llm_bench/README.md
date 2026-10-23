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
   - (optional) `--qps-distribution`: specify how to space out requests. Default is `constant` meaning evenly spaced out. `exponential` is an option simulating [Poisson distribution](https://en.wikipedia.org/wiki/Traffic_generation_model#Poisson_traffic_model). `gamma` draws gamma inter-arrival times with mean `1/qps` and squared coefficient of variation `--qps-cv2` (default 1, which is the same as `exponential`). `--qps-cv2` above 1 clumps arrivals into bursts like production traffic; below 1 spaces them more evenly. Clumped arrivals need more `-u` headroom than the average concurrency suggests.

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

Class-mix workload (`--shape-mix`), for traffic shaped like a production profile instead of a single prompt length with one shared prefix:
- `--shape-mix '<json>'` or `--shape-mix @profile.json`. The JSON is an object with `seed` (default 0), optional `arrivals.cv2` and a `classes` list; a traffic profile can be passed as is, other top-level fields are ignored. Each class has `weight`, `prompt_tokens`, `cached_prompt_tokens` (default 0), `generated_tokens`, `prefix_pool_size` (default 1, with a warning for classes that have cached tokens) and an optional `name`. Token counts may be fractional (profile means) and are rounded. Without `name`, the class is named from `cache_class` and `prompt_band` (`marginal` when both are null, `class<i>` when absent), made unique with a `_2`, `_3` suffix; names appear in the summary columns and the affinity key. Other unknown class fields are ignored with a warning.
- `arrivals.cv2` is the default for `--qps-cv2` when the flag isn't given (it only applies with `--qps-distribution gamma`).
- Each request picks a class by weight. Its prompt is one of the class's `prefix_pool_size` shared prefixes, `cached_prompt_tokens` long, followed by a unique tail up to exactly `prompt_tokens` (sized with `--tokenizer`, including the chat template in `--chat` mode). The tail starts with a per-request nonce, so a full prompt never repeats and a class with `cached_prompt_tokens: 0` stays cold.
- Requests that use a shared prefix send `x-session-affinity: shape-<seed>-<class>-<prefix index>`, so routing keeps each prefix on one replica as in production.
- Output length is forced per class: `max_tokens = min_tokens = generated_tokens` with `ignore_eos: true`.
- Prefixes depend only on `seed`, so every worker and every run uses the same prefix working set. The request sequence is reproducible for a fixed seed and worker; distributed workers get different tails.
- Replaces `-p`, `-o`, `--dataset` and `-pcml`, which can't be combined with it. Works with `--chat` and `--no-chat`, streaming or not.
- The summary adds, per class: `Class <name> Requests`, average `Time To First Token`, `Total Latency`, `Prompt Tokens`, `Cached Tokens` (as reported by the server) and `Completion Tokens`, plus P50/P90/P99 TTFT and latency.

```
locust -t 10min -u 200 -r 200 --qps 4 --qps-distribution gamma --qps-cv2 4 \
  --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct --summary-file results.csv \
  --shape-mix '{"seed": 0, "classes": [
    {"name": "chat", "weight": 0.7, "prompt_tokens": 6000, "cached_prompt_tokens": 5000, "generated_tokens": 200, "prefix_pool_size": 32},
    {"name": "rag", "weight": 0.2, "prompt_tokens": 12000, "cached_prompt_tokens": 2000, "generated_tokens": 400, "prefix_pool_size": 4},
    {"name": "cold", "weight": 0.1, "prompt_tokens": 2000, "generated_tokens": 100}]}'
```

Generation options:
- `--chat`: specify to call chat API instead of raw completions
- `--stream`: stream the result back. Enabling this gives "time to first token" and "time per token" metrics
- (optional) `--logprobs`: corresponds to `logprobs` API parameter. For some providers, it's needed for output token counting in streaming mode.

Embeddings and rerank options:
- `--embeddings`: use the `/v1/embeddings` API instead of completions
- `--rerank`: use the `/v1/rerank` API. The generated prompt text is split into documents (by paragraph), and `--rerank-query` is used as the query.
- `--rerank-query`: the search query string for rerank requests. Defaults to a generic query if not specified.
- `--rerank-top-n`: number of top results to return from the rerank endpoint.
- `--rerank-return-documents` / `--no-rerank-return-documents`: whether to include document text in the rerank response (default: true).

SystemOne options:
- `--systemone`: use the Fireworks `/v1/systemone` API (always non-streaming). `--model` is sent verbatim, e.g. `accounts/<account>/deployments/<id>`. Requests are synthetic: a JSON state built from the limericks dataset plus `q0..q{N-1}` choice questions with `option_a..` criteria.
- `--systemone-num-questions`: questions per request (default 1).
- `--systemone-num-options`: criteria per choice question (default 4).
- `--systemone-state-tokens`: approximate state size in tokens (default 768). Sized with `--tokenizer` if given, otherwise ~4 chars/token.
- `--systemone-question-tokens`: approximate instructions size per question in tokens (default 48).
- `--systemone-unique-state` / `--no-systemone-unique-state`: by default each state starts with a random nonce so every request misses the prefix cache; with `--no-systemone-unique-state` each worker reuses one fixed state (warm-cache path).

A request fails if the status isn't 200 or `answers` doesn't contain exactly the asked question ids. A response with `usage.output_tokens == 0` came from the packed single-pass path; the summary reports `Packed Fraction`, `Questions Per S` (QPS x questions), mean `Input Tokens` and total latency percentiles instead of generation metrics.

### Writing results

Locust prints out the detailed summary including quantiles of various metrics. Additionally, the script prints out the summary block at the very end of the output that includes the model being tested.

When comparing multiple configurations, it's useful to aggregate results together:

- `--summary-file`: Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, it writes out the header first. For Fireworks text runs, the summary also includes server-side time to first token and total latency (average and P50 to P99.9). Client-side values minus server-side values is roughly the network and gateway time between the load generator and the deployment.
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

Benchmark Fireworks SystemOne with 8 concurrent workers, 4 questions of 6 options each over a ~2k-token state:

```bash
locust -u 8 -r 8 -t 2min -H https://api.fireworks.ai/inference --api-key $FIREWORKS_API_KEY -m "accounts/pyroworks/deployments/pkrpl2xp" --systemone --systemone-num-questions 4 --systemone-num-options 6 --systemone-state-tokens 2048 --tokenizer /path/to/tokenizer --summary-file systemone_results.csv
```

Benchmark OpenAI deployment reading prompts from a file at 1 QPS:

```bash
locust --dataset '@input.jsonl' -u 1 -H https://api.openai.com -o 200 --api-key $OPENAI_API_KEY --model=gpt-3.5-turbo --chat
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
