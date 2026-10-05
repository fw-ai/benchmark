"""FAME v0 attention harness (CLI for nsys/ncu profiling)."""
import argparse, os, sys, json
from types import MethodType
import yaml
import torch
import torch.cuda.nvtx as nvtx
import fireworks.cuda as firecuda
import fireworks.nn as firenn
from fireworks.nn.attention.paged_attention import PrefillAttentionMeta, IncrAttentionMeta
from fireworks.nn.functional import apply_rope_qk
from fireworks.nn.rope_quantization import maybe_rope_quantize_paged_attn_qk_fp8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_layer import build_one_layer


def attention_forward_with_nvtx(self, x, rope, meta):
    bs = x.shape[0]
    nvtx.range_push("K1_qkv")
    qkv = self.query_key_value(x).view(bs, -1, self.head_dim)
    nvtx.range_pop()
    q, k, v = firenn.split_qkv(qkv, self.config.n_head_kv)
    def _qn():
        nvtx.range_push("K3_qnorm")
        out = self.q_norm(q.reshape(bs, -1), inplace=True).view(bs, -1, self.head_dim)
        nvtx.range_pop()
        return out
    def _kn():
        nvtx.range_push("K4_knorm")
        out = self.k_norm(k.reshape(bs, -1), inplace=True).view(bs, -1, self.head_dim)
        nvtx.range_pop()
        return out
    q2, k2 = firecuda.maybe_execute_in_parallel(_qn, _kn)
    q2 = q2[:, self._local_head_start:self._local_head_end]
    k2 = k2[:, self._local_kv_head_start:self._local_kv_head_end]
    v  = v [:, self._local_kv_head_start:self._local_kv_head_end]
    activation_dtype = q2.dtype
    self._rope_quant_scales, quantized_qk = maybe_rope_quantize_paged_attn_qk_fp8(
        attn=self.attn, rope_quant_scales=self._rope_quant_scales,
        positions=meta.positions, q=q2, k=k2, cos_sin_cache=rope,
        rope_dim=self.config.rope_dim, is_neox=True,
    )
    if quantized_qk is None:
        nvtx.range_push("K5_rope")
        qr = q2[..., :self.config.rope_dim]; kr = k2[..., :self.config.rope_dim]
        apply_rope_qk(meta.positions, qr, kr, rope, qr, kr)
        nvtx.range_pop()
    else:
        q2, k2 = quantized_qk
    nvtx.range_push("K6_paged_attn")
    y = self.attn(meta, qkv=None, rope=None, qk_scale=self.softmax_scale,
                  split_qkv=(q2, k2, v), activation_dtype=activation_dtype)
    nvtx.range_pop()
    y = y.flatten(-2, -1)
    nvtx.range_push("K7_oproj")
    out = self.o_proj(y).view(x.shape)
    nvtx.range_pop()
    return out


def build_inputs(block, hf_config, mode, p):
    cfg = block.self_attn.config
    n_head_kv = cfg.n_head_kv; head_dim = cfg.head_dim; n_embd = cfg.n_embd
    if mode == "prefill":
        T = p["prefill"]["seq_len"]; T_kv = T
    else:
        T = 1; T_kv = p["decode"]["past_kv"]
    block_size = p["paged_block_size"]
    blocks_per_seq = (T_kv + block_size - 1) // block_size
    num_seqs = 8
    kv = firenn.KVCache(
        config=firenn.KVCacheConfig(
            num_blocks=num_seqs * blocks_per_seq + 4,
            block_size=block_size,
            num_heads_kv=n_head_kv,
            head_size=head_dim,
            num_layers=1,
            cache_format="flat",
        ),
        dtype=torch.bfloat16, device="cuda",
    )
    if mode == "prefill":
        meta = PrefillAttentionMeta.empty_for_cuda_graph(kv, q_batch_size=T, max_kv_seq_len=T)
    else:
        meta = IncrAttentionMeta.empty_for_cuda_graph(kv, batch_size=1, max_seq_len=T_kv, num_timesteps=1)
    rope_cache = firenn.RoPECache(
        max_seq_length=cfg.max_seq_length, rope_dim=cfg.rope_dim,
        base=cfg.rope_theta, precision=torch.float32,
    )
    rope_cache.prepare_for_inference()
    rope = rope_cache().cuda()
    x = torch.randn(T, n_embd, dtype=torch.bfloat16, device="cuda") * 0.02
    return x, rope, meta, T, T_kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--multi-stream", choices=["on", "off"], default="on")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--measure", action="store_true")
    g.add_argument("--ncu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.multi_stream == "off":
        os.environ["FIREWORKS_DISABLE_MULTI_STREAM"] = "1"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    block, hf_config, fw_cfg = build_one_layer(cfg["weights_dir"], layer=cfg.get("layer", 0))
    attn = block.self_attn
    attn.forward = MethodType(attention_forward_with_nvtx, attn)
    p = cfg["profile"]
    x, rope, meta, T, T_kv = build_inputs(block, hf_config, args.mode, p)
    with torch.no_grad():
        if args.check:
            y = attn(x, rope, meta)
            assert y.shape == x.shape and torch.isfinite(y).all()
            print(f"correctness OK: mode={args.mode} T={T} T_kv={T_kv} out_shape={tuple(y.shape)}")
            return
        if args.ncu:
            attn(x, rope, meta)
            torch.cuda.synchronize()
            return
        for _ in range(p["warmup"]):
            attn(x, rope, meta)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for _ in range(p["measured"]):
            nvtx.range_push("iter")
            attn(x, rope, meta)
            nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    out = args.out or f"workspace/{args.mode}_manifest.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mode": args.mode, "T": T, "T_kv": T_kv,
                   "iters": p["measured"], "warmup": p["warmup"],
                   "multi_stream": args.multi_stream,
                   "weights_dir": cfg["weights_dir"], "layer": cfg.get("layer", 0)}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
