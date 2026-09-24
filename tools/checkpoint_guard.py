#!/usr/bin/env python3
"""Checkpoint guard for the GLM-5.3-Flash launchers.

usage: checkpoint_guard.py <model_dir> dflash2|mtp
Exit 0 = OK to launch, 5 = refuse (reason on stderr).

dflash2: ModelOpt NVFP4 builds are allowed only in the NVIDIA shape, where every layer's
         whole self_attn block is excluded from quantization (nvidia/GLM-5.3-Flash-NVFP4:
         132-entry ignore list, clean on the Hangul probe, issue #23). Partial-attention
         ModelOpt builds (e.g. LibertAIDAI, 4/9/8 U+FFFD, vLLM #54150) are refused.
         compressed-tensors (RedHatAI) is always allowed. ALLOW_MODELOPT=1 skips the check.
mtp:     additionally refuses a checkpoint whose MTP head (layer num_hidden_layers) is stored
         in a different quantization than the layer before it (nvidia ships layer 45 in BF16
         while its ignore list stops at 44, so the draft MoE cannot load, and no sm121 MoE
         backend serves an NVFP4 target plus an unquantized draft MoE; issue #23).
"""
import fnmatch
import json
import os
import sys


def refuse(msg):
    print("REFUSING: " + msg, file=sys.stderr)
    sys.exit(5)


def main():
    model_dir, mode = sys.argv[1], sys.argv[2]
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return 0  # launcher reports the missing weights itself
    cfg = json.load(open(cfg_path))
    q = cfg.get("quantization_config") or {}
    tc = cfg.get("text_config") or cfg
    n = int(tc.get("num_hidden_layers") or 0)
    method = q.get("quant_method", "")

    if method == "modelopt" and os.environ.get("ALLOW_MODELOPT", "0") != "1":
        ignore = q.get("ignore") or q.get("exclude_modules") or []
        probe = "model.language_model.layers.{}.self_attn.__whole_block__"
        missing = [i for i in range(n) if not any(fnmatch.fnmatch(probe.format(i), p) for p in ignore)]
        if missing:
            refuse(f"{model_dir} is a ModelOpt build that quantizes attention in {len(missing)} of {n} layers "
                   "(not the nvidia shape). These builds emit intermittent corrupted token IDs on this stack "
                   "(vLLM #54150). Use nvidia/GLM-5.3-Flash-NVFP4 or RedHatAI/GLM-5.3-Flash-NVFP4, "
                   "or set ALLOW_MODELOPT=1 to override.")

    if mode == "mtp" and n:
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        try:
            names = json.load(open(idx_path)).get("weight_map", {}).keys()
        except (OSError, ValueError):
            names = None  # no readable index: the check is skipped, never a false refusal
        if names:

            def scaled(layer):
                tag = f".layers.{layer}.mlp.experts."
                return any(tag in k and "scale" in k for k in names)

            def present(layer):
                tag = f".layers.{layer}.mlp.experts."
                return any(tag in k for k in names)

            if present(n) and scaled(n) != scaled(n - 1):
                refuse(f"{model_dir}: the MTP head (layer {n}) is stored "
                       f"{'quantized' if scaled(n) else 'unquantized'} while layer {n - 1} is "
                       f"{'quantized' if scaled(n - 1) else 'unquantized'}. MTP cannot be served on "
                       "sm121 with this checkpoint (see issue #23); use the DFlash2 launcher instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
