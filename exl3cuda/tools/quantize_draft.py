#!/usr/bin/env python3
"""Build a cheaper MTP draft package: re-quantize the MTP layer's linears and a draft-vocab
subset of lm_head from the original BF16 weights to EXL3 at a lower bitrate. Drafts only: the target model is unchanged,
so the output distribution stays exact; only acceptance can change. On the 5090 a draft pass reads ~465 MB (MTP layer
213 MB at 4 bpw + a 512-block draft head 252 MB at the checkpoint's 6 bpw), ~2.8 GB per step at k=6.

No calibration Hessian (exllamav3's fallback path, H on the meta device); seed fixed. Output: one safetensors file with
  mtp.<module>.{trellis,suh,svh,mul1}   for fc, self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj
  draft_head.{trellis,suh,svh,mul1,blocks}   lm_head rows of the vocab blocks (128-token blocks, the EXL3_DRAFT_VOCAB json)
Plugin: EXL3_DRAFT_WEIGHTS=<file> (exl3cuda/vllm_plugin.py) swaps them in at load time.
  quantize_draft.py --bf16 <dir with the shard holding mtp.* and lm_head> --vocab draft_vocab.json --k-mtp 3 --k-head 4 --out f
"""
import argparse, json, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

ap = argparse.ArgumentParser()
ap.add_argument("--bf16", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--k-mtp", type=int, default=3)
ap.add_argument("--k-head", type=int, default=4)
ap.add_argument("--out", required=True)
ap.add_argument("--no-mtp", action="store_true", help="head only: keep the checkpoint's MTP layer (plugin keeps unlisted modules)")
ap.add_argument("--hessian", default=None, help="dir of captured X^T X (EXL3_CAPTURE_MTP_H) for calibrated quantization")
a = ap.parse_args()

idx = json.load(open(f"{a.bf16}/model.safetensors.index.json"))["weight_map"]


def get(name):
    with safe_open(f"{a.bf16}/{idx[name]}", "pt", device="cuda") as f:
        return f.get_tensor(name)


import os
H_FOR = {"fc": "mtp.fc", "q_proj": "mtp.layers.0.self_attn.qkv_proj", "k_proj": "mtp.layers.0.self_attn.qkv_proj",
         "v_proj": "mtp.layers.0.self_attn.qkv_proj", "o_proj": "mtp.layers.0.self_attn.o_proj",
         "gate_proj": "mtp.layers.0.mlp.gate_up_proj", "up_proj": "mtp.layers.0.mlp.gate_up_proj",
         "down_proj": "mtp.layers.0.mlp.down_proj", "draft_head": "draft_head"}


def hdata(name, k):
    """Captured Hessian for a module (fresh dict per tensor), or the uncalibrated fallback."""
    if a.hessian:
        f = os.path.join(a.hessian, H_FOR[name.split(".")[-1]] + ".pt")
        if os.path.exists(f):
            d = torch.load(f)
            assert d["H"].shape[0] == k, (name, d["H"].shape, k)
            return {"H": d["H"].cuda(), "count": d["count"], "device": torch.device("cuda"), "finalized": False,
                    "first_key": name}, d["count"]
        print(f"  (no captured Hessian for {name}: uncalibrated)")
    return {"H": torch.empty(k, k, device="meta"), "device": torch.device("cuda"), "finalized": False}, 0


def quant(w_out_in, K, seed, name):
    w = w_out_in.float().T.contiguous()                     # quantize_exl3 wants (in, out); it modifies its input
    k = w.shape[0]
    Hd, cnt = hdata(name, k)
    qa = {"K": K, "seed": seed, "devices": [0], "mul1": True, "apply_out_scales": None, "sigma_reg": 0.025}
    _, perr, out = quantize_exl3(w, Hd, qa, return_weight_q=False)
    return {kk: v.cpu() for kk, v in out.items()}, perr


tensors = {}
mods = ["fc", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj", "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "layers.0.mlp.down_proj"]
for i, m in enumerate([] if a.no_mtp else mods):
    t0 = time.time()
    out, perr = quant(get(f"mtp.{m}.weight"), a.k_mtp, 1000 + i, m)
    for kk, v in out.items():
        tensors[f"mtp.{m}.{kk}"] = v
    print(f"mtp.{m}: K={a.k_mtp} proxy err {perr:.5f} ({time.time() - t0:.0f} s)", flush=True)

spec = json.load(open(a.vocab))
bs = spec["block_size"]; assert bs % 128 == 0
blocks = torch.tensor(spec["blocks"], dtype=torch.long)
rows = (blocks[:, None] * bs + torch.arange(bs)[None, :]).flatten()
W = get("lm_head.weight")                                   # (vocab, hidden)
rows = rows[rows < W.shape[0]]
t0 = time.time()
out, perr = quant(W.index_select(0, rows.cuda()), a.k_head, 77, "draft_head")
for kk, v in out.items():
    tensors[f"draft_head.{kk}"] = v
tensors["draft_head.blocks"] = blocks.to(torch.int32)
tensors["draft_head.block_size"] = torch.tensor(bs, dtype=torch.int32)
print(f"draft_head: {len(rows)} rows, K={a.k_head} proxy err {perr:.5f} ({time.time() - t0:.0f} s)", flush=True)
save_file({k: v.contiguous() for k, v in tensors.items()}, a.out,
          metadata={"k_mtp": "ckpt" if a.no_mtp else str(a.k_mtp), "k_head": str(a.k_head), "vocab": a.vocab, "source": a.bf16})
mb = sum(v.numel() * v.element_size() for v in tensors.values()) / 1e6
print(f"wrote {a.out}: {mb:.0f} MB", flush=True)
