# FAME v0 — MiniMax M2.7 attention roofline (measured)

FLOPs, bytes, and %peak from Nsight Compute hardware counters. Timings from Nsight Systems.

## Prefill (seq_len=4096)

| kernel | nsys_us | ncu_dur_us | FLOPs_GF | bytes_MB | AI | %peak_compute | %peak_BW | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| K1_qkv | 119.8 | 242.2 | 154.62 | 151.25 | 1022.3 | 86.4 | 51.3 | compute |  |
| K3_qnorm | 19.3 | 32.8 | 0.03 | 52.41 | 0.5 | 47.0 | 43.6 | compute |  |
| K4_knorm | 6.2 | 12.8 | 0.01 | 8.40 | 1.0 | 18.1 | 19.2 | memory |  |
| K5_rope | 13.5 | 0.0 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | latency | fused into K6 |
| K6_paged_attn | 97.0 | 87.0 | 274.88 | 0.09 | 3112295.1 | 27.5 | 15.6 | compute |  |
| K7_oproj | 90.4 | 157.5 | 103.08 | 166.95 | 617.4 | 76.8 | 38.2 | compute |  |

## Decode (past_kv=4096)

| kernel | nsys_us | ncu_dur_us | FLOPs_GF | bytes_MB | AI | %peak_compute | %peak_BW | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| K1_qkv | 8.4 | 19.2 | 0.04 | 50.35 | 0.7 | 24.4 | 39.3 | memory |  |
| K3_qnorm | 2.2 | 7.9 | 0.00 | 0.03 | 0.2 | 0.0 | 2.1 | latency |  |
| K4_knorm | 1.7 | 6.8 | 0.00 | 0.01 | 0.2 | 0.0 | 2.5 | latency |  |
| K5_rope | 2.0 | 0.0 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | latency | fused into K6 |
| K6_paged_attn | 4.9 | 14.8 | 0.07 | 0.06 | 1036.1 | 4.7 | 2.8 | latency |  |
| K7_oproj | 9.0 | 16.6 | 0.03 | 37.78 | 0.7 | 19.7 | 34.0 | memory |  |