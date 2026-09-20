# Experimental: NVFP4 the attention and MLP projections (untested at TP2)

**Status: experimental. Measured at TP4 only. It MAY give a similar speed increase at TP2, and it has not been
tested on this lane. Nothing in this document is a TP2 measurement.**

Measured at TP4 on 2026-09-20 and written up in full in the sibling repo:
[GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark, `runs/2026-09-20-tp4-vs-deepseek`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/tree/main/runs/2026-09-20-tp4-vs-deepseek).
The conversion and build tooling lives in that run's `tools/` (`mknvfp4b.py`, `build4.py`) and the two patched
model files in its `patches/`.

## The finding

The NVFP4 packs quantize the **routed experts** and leave everything else in bf16. Summing this checkpoint class
by module and dtype:

| module class | GiB | dtype |
|---|---|---|
| routed experts | 163.27 | U8 145.12 + F8_E4M3 18.14 |
| **attention** | **11.70** | **BF16** |
| dense MLP | 3.52 | BF16 |
| embeddings + head | 2.37 | BF16 |
| other | 0.43 | BF16 |

That is **18.01 GiB of non-expert weights, all bf16, read on every decode step** regardless of how many tokens
are in flight. Only 13.88 GiB of it is safe to quantize: `embed_tokens` and `lm_head` are a lookup and the logit
projection, the DSA `indexer.*` weights choose which KV blocks get attended, `mlp.gate` chooses experts, and the
KDA `f_b`/`g_b` gates are gating rather than projection. Those stay bf16.

## Why it should transfer to TP2, with the arithmetic

The attention and MLP projections are **sharded across ranks**, so halving the rank count doubles the bytes each
rank reads per step. On GB10 at 273 GB/s:

| | TP4 (measured) | TP2 (predicted) |
|---|---|---|
| bf16 non-expert per rank | 4.50 GiB -> 17.7 ms | 9.01 GiB -> **35.4 ms** |
| share of the decode step | 26% of 67.6 ms | **31% of ~116 ms** |
| routed experts at 8 verify tokens | 9.07 GiB -> 35.7 ms | 18.14 GiB -> 71.4 ms |
| NVFP4 saves, per rank | 2.60 GiB -> 10.2 ms | **5.21 GiB -> 20.5 ms** |
| predicted speedup | x1.18 (measured x1.14-1.30) | **x1.21** |
| weight residency freed | 2.60 GiB/rank | **5.21 GiB/rank** |

The TP2 step figure is derived from the 2026-09-18 healthy-fleet numbers in `speed-night-2026-09-18`: code at
44.7 tok/s with roughly 5.2 accepted tokens per step gives a ~116 ms step, and the cost model built at TP4
(`step_ms = 34.6 + 4.12 x verify_tokens`, residuals under 2.2 ms over four draft lengths) reproduces that to
within about 7% when rescaled to TP2. So the model transfers; the experiment has not been run.

**The memory saving may matter more at TP2 than the speed.** Freeing 5.21 GiB/rank on a 121 GiB node is directly
relevant to the two levers that did not fit on 2026-09-18: `seqs 32 + mnbt 16384` died with
`NVRM: NV_ERR_NO_MEMORY` at the first real batch, and the KV pin at 6 -> 8 GiB was worth +33% of pool. Both were
memory-bound, and this hands back more than the KV bump did.

## Why it needs a patch, not just a repacked checkpoint

`glm5next/nvidia` hardcodes those projections as bf16, independent of what the checkpoint declares:

```
kda.py:172     vllm_config.quant_config = None            # strips quant for the whole KDA submodule tree
model.py:331   quant_config=None,  # MLA projections are BF16 in checkpoint
```

A packed NVFP4 `(out, in/2)` weight cannot load into a bf16 `(out, in)` parameter, so the loader asserts. Four
boots failed on this at TP4 before the constructor was read rather than the loader. The fix is two bind-mounted
lines behind an opt-in `NVFP4_PATCH=1`, which leaves the default lane untouched. Left alone deliberately:
`model.py:1090` (vision tower, quantizing it yields NaN image features) and `attention.py:263` (indexer
`wk_weights_proj`).

Selecting `quant_algo: "W4A16_NVFP4"` instead of `"NVFP4"` is required for the weight path and **also disarms a
W4A4 landmine** in this checkpoint class: it declares `NVFP4` while shipping zero `input_scale` tensors, so vLLM
creates them with `torch.empty` and any W4A4 backend multiplies by uninitialised memory. Marlin escapes by
discarding those scales.

## What to run to test it here

1. Build the checkpoint variant with the TP4 repo's `mknvfp4b.py` then `build4.py`. Note `build4.py` **rewrites
   the shards that mix superseded bf16 tensors with tensors still needed** - the safetensors index filters by
   file, not tensor name, so hardlinking the originals and retargeting the index is not sufficient and will load
   the stale bf16 copies.
2. Distribute the two patched files to both nodes of the lane and boot with `NVFP4_PATCH=1`.
3. Measure with `bench_tp2_night.py --suite all` against the shipped recipe, medians of 3. Single-run cells on
   these lanes carry a plus-or-minus 25% band, so single-pass comparisons will not resolve a 20% effect.
4. Then retest `SEQS=32` and the KV ladder, since the 5.21 GiB/rank is the point of interest at TP2.

## Caveats carried over from TP4

- **Refusal behaviour is untested.** In the abliterated packs the modification is 31
  `layers.{15..45}.self_attn.o_proj.weight` tensors, and `o_proj` is the largest single component of what gets
  quantized. Whether abliteration survives quantization unchanged is unknown in either direction.
- **Long context: verified to 450K at a 500K window, and one needle dropped a digit at 131K on a 1M window.**
  Same checkpoint, same quantized tensors, so the fault tracked `max_model_len` rather than the quantization, but
  the mechanism is unconfirmed.
- **Image drift.** The patch bind-mounts copies of two files extracted from a specific image. If the image is
  rebuilt, those copies silently override newer versions. Pin the image digest alongside the patch.
- Concurrency above C3 was not measured at TP4 when this was written.
