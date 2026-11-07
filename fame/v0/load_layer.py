"""Load one MiniMax M2.7 transformer Block with real FP8 weights for layer L."""
import os, json
import torch
from safetensors import safe_open
from transformers import AutoConfig
import fireworks.nn as firenn
from fireworks.models.minimax_m2.minimax_m2 import Block
from fireworks.models.minimax_m2.minimax_m2_text import MiniMaxM2CausalTextModel


def _index_layer_files(weights_dir, layer):
    idx = json.load(open(os.path.join(weights_dir, "model.safetensors.index.json")))
    prefix = f"model.layers.{layer}."
    files = set()
    for k, fn in idx["weight_map"].items():
        if k.startswith(prefix):
            files.add(os.path.join(weights_dir, fn))
    return files


def _load_layer_state(weights_dir, src_layer):
    files = _index_layer_files(weights_dir, src_layer)
    src_prefix = f"model.layers.{src_layer}."
    dst_prefix = "layers.0."
    out = {}
    for f in sorted(files):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                if not k.startswith(src_prefix):
                    continue
                out[dst_prefix + k[len(src_prefix):]] = h.get_tensor(k)
    return out


def build_one_layer(weights_dir, layer=0):
    hf_config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    hf_config.num_hidden_layers = 1
    hf_config.use_mtp = False
    fw_cfg = MiniMaxM2CausalTextModel.convert_hf_config(hf_config)
    with torch.device("cuda"):
        model = MiniMaxM2CausalTextModel(fw_cfg)
    raw = _load_layer_state(weights_dir, src_layer=layer)
    fw_state = MiniMaxM2CausalTextModel.convert_hf_base_state_dict(
        fw_cfg, {"model." + k: v for k, v in raw.items()}
    )
    missing, unexpected = model.load_state_dict(fw_state, strict=False)
    block = model.layers[0]
    block_param_names = {f"layers.0.{n}" for n, _ in block.named_parameters()}
    real_missing = [k for k in missing if k in block_param_names]
    if real_missing:
        raise RuntimeError(f"Missing block weights ({len(real_missing)}): {real_missing[:10]}")
    block.eval().requires_grad_(False)
    print(f"loaded Block from {weights_dir} layer={layer}")
    print(f"  block params: {sum(p.numel() for p in block.parameters())/1e9:.2f} B")
    print(f"  cuda mem    : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return block, hf_config, fw_cfg


if __name__ == "__main__":
    import sys
    block, hf, fw = build_one_layer(os.environ["WEIGHTS_DIR"],
                                     layer=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print("OK")
