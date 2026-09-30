# GLM-5.3-Flash NVFP4 + DFlash2 on 2x NVIDIA DGX Spark

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-tp2/charts/tp2prompt-dark.svg">
  <img alt="Single-stream decode by prompt type on 2x DGX Spark TP2, previous recipe vs knapcio stack: count to 100 61 to 96, count to 300 58 to 93, tool call 48 to 101, math 46 to 65, code 45 to 69, json 43 to 71, sql 40 to 60, summary 21 to 42, prose 18 to 35, narrative 18 to 35 tok/s." src="runs/2026-09-29-knapcio-tp2/charts/tp2prompt-light.svg" width="880">
</picture>

**Single-stream decode, tok/s** (median of 3). Same fleet, same harness, same nvidia weights. Count to 100 is the peak.

| prompt | previous TP2 recipe | **knapcio stack, lane A (thinking low)** | change | lane B (thinking high) |
|---|---|---|---|---|
| count to 100 (peak) ¹ | 61.3 | **96.4** | +57% | 92.5 |
| count to 300 ¹ | 57.6 | **93.0** | +61% | 87.5 |
| tool call | 47.7 | **101.1** | +112% | 66.9 |
| math | 46.2 | **65.4** | +42% | 65.5 |
| code | 44.7 | **68.9** | +54% | 63.8 |
| json | 42.8 | **71.0** | +66% | 70.6 |
| sql | 39.6 | **60.3** | +52% | 59.7 |
| summary | 20.9 | **41.8** | +100% | 40.7 |
| prose | 18.1 | **35.3** | +95% | 35.5 |
| narrative | 17.9 | **35.3** | +97% | 34.6 |

¹ The counting prompts are the speculative-decoding ceiling, not a typical rate. Previous recipe measured 2026-09-18 on the healthy fleet; knapcio 2026-09-29 at the default 6 GiB KV pin.

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active MoE) served by vLLM at **tensor-parallel 2 across two DGX Spark** (GB10/SM121), **262,144-token context**, fp8 KV, DFlash2 speculative drafter.

**Current default (2026-09-29): [knapcio's stack](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4), ported to two Sparks.** Decode is 1.4x to 2.1x faster on every prompt type than our previous recipe, C1 to C5 aggregate +19% to +62% (C6 even), weights load in 44 s, and the default 6 GiB KV pin fits two full 262K requests. Runbook and configs: [`runs/2026-09-29-knapcio-tp2/`](runs/2026-09-29-knapcio-tp2/).

**Previous recipe: [CURRENT.md](CURRENT.md)**, one launcher, [`launch-glm53-vllm-tp2-dflash2.sh`](launch-glm53-vllm-tp2-dflash2.sh) (worker Spark4 rank 1 first, then head Reddie rank 0, serving :8000). Still valid, and the fallback when you need its bigger KV pool (714K vs 560K tokens) or faster prefill.

Weights: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) at `/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia` for the DFlash2 launcher (RedHatAI at `/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat` for MTP or when memory is tight). ModelOpt builds that quantize attention, including the abliterated ones, corrupt tokens on this stack and the launchers refuse them.

Everything else here is reference: the bring-up log, the day-0 bug receipts, the benchmark history, and the open problems.

---

## ⭐ Current default (2026-09-29): knapcio's stack, ported to TP2

The speed stack is **[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)** by
[@knapcio](https://github.com/knapcio) (MIT), pinned at `770d115`. He ships it for **four** Sparks only; this repo ports it
to two: a four-line change to his launcher, two lane configs, and TP2 memory sizing. His stack is built on this repo
family's v11 image, RoCE all-reduce port and DFlash2 prefix-cache repair, and its largest single win (8-bit dense
layers) starts from our finding that the nvidia checkpoint leaves 18 GiB in BF16. The TP4 sibling runs it unmodified:
[GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark).

**Runbook (what the port changes and why), launcher, lane configs, harness and raw results: [`runs/2026-09-29-knapcio-tp2/`](runs/2026-09-29-knapcio-tp2/).**

| lane | nodes | port | default reasoning effort | KV pool | boot to `/health` |
|---|---|---|---|---|---|
| **A** | Reddie (head) + Spark4 | `:8000` | **low** | 560,362 tokens (6 GiB/rank) | about 170 s |
| **B** | Bluey (head) + Asusi | `:8001` | high | 560,362 tokens (6 GiB/rank) | about 190 s |

Both lanes serve `glm-5.3-flash` at 262K context and run identical code; only the server's default `reasoning_effort`
differs, and a request can override it. Weights load in **44 s** (knapcio's fast loader); the previous recipe took about
11 minutes. Two full 262K requests fit in the pool at once.

### C1 to C6, mixed real prompts

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-tp2/charts/tp2sweep-dark.svg">
  <img alt="Aggregate throughput C1 to C6 on TP2, peak of 3 rounds: previous recipe 46 / 52 / 58 / 60 / 66 / 66, knapcio lane A 67 / 84 / 73 / 72 / 82 / 66 tok/s." src="runs/2026-09-29-knapcio-tp2/charts/tp2sweep-light.svg" width="880">
</picture>

| streams | previous TP2 recipe | **knapcio lane A (low)** | change (peak) | knapcio lane B (high) |
|---|---|---|---|---|
| C1 | 42.7 / 46.1 | 63.3 / **67.3** | +46% | 63.8 / 64.9 |
| C2 | 33.8 / 52.0 | 57.2 / **84.0** | +61% | 59.4 / 82.1 |
| C3 | 35.9 / 57.7 | 46.4 / **73.4** | +27% | 59.9 / 86.9 |
| C4 | 59.0 / 60.5 | 71.5 / **72.1** | +19% | 87.7 / 88.7 |
| C5 | 40.9 / 66.3 | 46.3 / **81.9** | +24% | 71.9 / 96.7 |
| C6 | 51.6 / 65.8 | 65.7 / **65.7** | -0% | 80.1 / 81.6 |

Aggregate tok/s, median / peak of 3 rounds; 8 prompt types rotated across streams, no counting. Zero failures, zero
preemptions.

### Cold prefill and long context

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-tp2/charts/tp2prefill-dark.svg">
  <img alt="Cold prefill on TP2 at 6 GiB KV: previous recipe 1506 / 1242 / 1336, knapcio lane A 1121 / 1200 / 1225 tok/s." src="runs/2026-09-29-knapcio-tp2/charts/tp2prefill-light.svg" width="880">
</picture>

| | previous TP2 recipe | **knapcio lane A** | change |
|---|---|---|---|
| 5,944-token prompt | 1,506 tok/s (TTFT 3.9 s) | **1,121 tok/s** (TTFT 5.3 s) | -26% |
| 29,867-token prompt | 1,242 tok/s (TTFT 24.1 s) | **1,200 tok/s** (TTFT 24.9 s) | -3% |
| 113,911-token prompt | 1,336 tok/s (TTFT 85.3 s) | **1,225 tok/s** (TTFT 93.0 s) | -8% |
| 114K-token prompts, 1 at once: decode per stream | 19.1 tok/s | **26.9 tok/s** (lane B: 41.1) | +41% |
| 114K-token prompts, 2 at once: decode per stream | 9.2 tok/s | **14.0 tok/s** (lane B: 22.7) | +52% |

Prefill is flat to slower at TP2: knapcio's prefill speedups (mHC prefill sharding, routed-MoE prefill kernels) are built
for four ranks and are off in the port, and the 6 GiB pin leaves less memory headroom (next table). Long-context rows are
single runs.

### 6 GiB (default) or 4 GiB

Same stack, same day, only the KV pin changed. 6 GiB buys a 50% bigger pool; 4 GiB leaves more memory free, which shows
up in prefill and, on lane A (whose head also serves the weights to its worker over NFS), in concurrency.

| | 4 GiB pin | **6 GiB pin (default)** |
|---|---|---|
| KV pool | 372,773 tokens (1 full 262K request) | **560,362 tokens (2 full 262K requests)** |
| free memory, tightest node, under load | 3 GiB | 1 to 2 GiB |
| single-stream decode, code / json / prose (lane A) | 71.0 / 71.3 / 35.6 | 68.9 / 71.0 / 35.3 |
| cold prefill 6K / 30K / 114K (lane A) | 1,288 / 1,455 / 1,479 | 1,121 / 1,200 / 1,225 |
| C3 to C6 aggregate peak, lane A | 92 / 92 / 97 / 78 | 73 / 72 / 82 / 66 |
| C3 to C6 aggregate peak, lane B | 90 / 92 / 100 / 76 | 87 / 89 / 97 / 82 |
| 114K-token prompt, decode (lane A, single run) | 40.7 | 26.9 |

Pick 4 GiB (`KV_BYTES=4294967296`) when you never need two long contexts at once and want the faster prefill.

### Thinking low vs high (lane A vs lane B)

| prompt | decode, low | decode, high | first answer token, low | first answer token, high | thinking text, low | thinking text, high |
|---|---|---|---|---|---|---|
| count to 100 | 96.4 | 92.5 | 0.43 s | 0.53 s | 27 chars | 46 chars |
| count to 300 | 93.0 | 87.5 | 0.47 s | 0.77 s | 27 chars | 105 chars |
| tool call | 101.1 | 66.9 | 0.51 s | 0.52 s | 0 chars | 24 chars |
| math | 65.4 | 65.5 | 1.98 s | 3.86 s | 199 chars | 486 chars |
| code | 68.9 | 63.8 | 0.38 s | 1.26 s | 0 chars | 134 chars |
| json | 71.0 | 70.6 | 0.39 s | 0.54 s | 0 chars | 39 chars |
| sql | 60.3 | 59.7 | 0.40 s | 1.75 s | 0 chars | 246 chars |
| summary | 41.8 | 40.7 | 0.33 s | 1.29 s | 0 chars | 186 chars |
| prose | 35.3 | 35.5 | 0.31 s | 2.88 s | 0 chars | 382 chars |
| narrative | 35.3 | 34.6 | 0.33 s | 0.99 s | 0 chars | 147 chars |

Decode speed is about the same on most prompts. Tool call and code are slower at high because the thinking text accepts
fewer draft tokens than the answer does. The bigger difference is the wait: high thinks several times longer before it
answers, so the first answer token arrives later (up to 3.9 s on math, 2.9 s on prose). The TP4 sibling's
69-scenario tool-calling eval scored low 93.5 vs high 92.8, which is why low is the fleet default.

### What to know before you switch

- **This is our port, not knapcio's release.** He has not run TP2. Three of his pieces are off because they are built
  for four ranks (mHC prefill sharding refuses to boot otherwise, the routed-MoE prefill kernels, the prefill gather
  route); everything on the decode side is on and armed.
- **Memory is tight.** 87.2 GiB of weights per rank. KV pins tried: 8 GiB left 1 to 2 GiB free at idle; **7 GiB ran out of
  GPU memory on a 110K-token prompt** (`NV_ERR_NO_MEMORY`, no crash); **6 GiB passes** one 111K prompt (3/3 needles) and two
  concurrent 114K prompts, with the tightest node at 1 to 2 GiB free; 4 GiB is the extra-margin option. The 6 GiB config
  also trims the image budget to 2 images per prompt.
- **Censored checkpoint** (`nvidia/GLM-5.3-Flash-NVFP4`, which is also this repo's previous default).
- **Thinking cannot be switched off** in his chat template (`reasoning_effort` low, high or max).
- **Single RoCE rail**, same as the TP4 sibling.

---

## Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` is the DFlash2 default (2026-09-24, issue #23)

ModelOpt NVFP4 builds that quantize attention (`LibertAIDAI/GLM-5.3-Flash-NVFP4` and the abliterated variants) emit **intermittent corrupted token IDs** ([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)). Nearly invisible in English, but when a corrupted token lands inside a tool-call block the parser desyncs and generation can spiral into a repetition lock. NVIDIA's own build keeps every layer's attention in high precision and is clean.

Korean-Hangul probe (`temperature 0`, non-streaming, 3 passes):

| checkpoint | `quant_method` | attention | U+FFFD count (3 runs) |
|---|---|---|---|
| ModelOpt NVFP4 (LibertAIDAI / keys-ablit) | `modelopt` | partly quantized | 4 / 9 / 8 |
| **[nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)** | `modelopt` | **all 45 `self_attn` blocks excluded** (132-entry ignore list) | **0 / 0 / 0** ([#23](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/issues/23), @calvarado2004) |
| [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | `compressed-tensors` | not quantized | 0 / 0 / 0 |

**Default for the DFlash2 launcher: `nvidia/GLM-5.3-Flash-NVFP4`** at `/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia` (the launcher falls back to the RedHat path if only that copy is on disk). Why: only the routed experts and the 3 dense MLP layers are NVFP4 and activations stay 16-bit (W4A16), where RedHatAI is W4A4. It is also the build our TP4 recipe runs as its censored default, and the 2026-09-18 speed night measured this launcher's config on it (8 GiB KV pin: 714,240 tokens at 262K). Tradeoff: it holds about 3 GiB/rank more than RedHatAI (132 bf16 modules), so RedHatAI stays the pick when memory is the constraint. DFlash2 k=7 on it, measured in #23: code 51.9, count 42.5, math 34.9, prose 28.4 tok/s (spec off: prose 14.1, code 14.0).

**MTP does not work on the nvidia build.** NVIDIA ships the MTP head (layer 45) in BF16 while its ignore list stops at layer 44, so the draft MoE fails to load, and even with the ignore list fixed no sm121 MoE backend serves an NVFP4 target plus an unquantized draft MoE (`marlin` refuses unquantized MoE, `triton` refuses NVFP4, `flashinfer_trtllm` is sm100-only, `flashinfer_cutlass` cannot JIT in this image). DFlash2's drafter is a small dense model, so it is unaffected. The MTP launchers (`launch-glm53-vllm-tp2.sh`, `launch-glm53-vllm-tp4.sh`) keep RedHatAI as their checkpoint and now refuse a checkpoint whose MTP head is stored in a different quantization than the model. Full analysis: #23.

`tools/checkpoint_guard.py` runs in every launcher: it refuses ModelOpt builds that quantize attention (override `ALLOW_MODELOPT=1`) and, for the MTP launchers, mismatched MTP heads. Make sure the vision `chat_template_mm.jinja` is present in the weights dir or image requests 500.

Corruption first flagged by [@ajclark](https://github.com/ajclark) (issue #10). Uncensored (abliterated) builds remain available but carry the ModelOpt corruption until a clean abliteration exists (the TP4 recipe runs `Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4` with a config fix; it needs `ALLOW_MODELOPT=1` here).

Image notes from #23: `flashinfer_cutlass` MoE cannot JIT in `sm121-v11-dflash2` (`nvrtc.h` is in the image but off the include path; putting the whole `nvidia/cu13/include` on `CPATH` breaks the CUTLASS stubs). One `cicc` of that FP4 build reached 8.7 GB RSS: warm such caches with `MAX_JOBS=1` beside a loaded model. KV bytes per token depend on the speculative config (6,996 without a drafter, 9,178 with DFlash2 k=7), so size `--kv-cache-memory` with the drafter on.

## Weights: censored or uncensored (drop-in)

Pick your weights: **same launcher, same recipe**, just point the model path at either. Both are NVFP4 and load identically.

| | HuggingFace | notes |
|---|---|---|
| **⭐ Default (DFlash2)** | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | ModelOpt W4A16, attention kept in high precision, corruption-free; no MTP (see above) |
| Memory-lean / MTP | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | compressed-tensors W4A4, corruption-free, about 3 GiB/rank smaller; the MTP launchers' checkpoint |
| Censored (legacy) | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) | stock NVFP4 weight-only — ⚠️ ModelOpt token corruption |
| **Uncensored (abliterated)** | [drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock) | abliterated (layers 15-45, anchor-stock), no refusals |

Uncensored abliteration credit: [drowzeys/keys](https://github.com/drowzeys).

Two firsts, as far as we can tell: the **first working GLM-5.3-Flash deployment on DGX Spark**
(seven day-0 bugs deep — [docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md)), and the **first
working DFlash2 deployment of this model on GB10** ([docs/DFLASH2-SPECULATIVE-DECODING.md](docs/DFLASH2-SPECULATIVE-DECODING.md)).

> 🔀 **Running all four Sparks?** The same images scale to TP4 with the model-native 1M context —
> see the sibling repo: **[GLM-5.3-Flash at TP4 · 1M KV · 4x DGX Spark →](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)**

---

## Quickstart

**1. Pull the images** (GHCR, public, anonymous — no build required):

```bash
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2   # with DFlash2 (recommended)
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8            # base, fp8 KV, no drafter
```

They contain **only vLLM + our patches** — no model weights (those bind-mount at runtime).
No retag step: the launchers reference these `ghcr.io/tonyd2wild/…` tags directly. (Only the
`docker/` build chain uses local `radixark/…` stage tags, and those never leave the build.)

**2. Fetch the weights** to the same path on both nodes (or NFS-export from the head):
[nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (DFlash2 default) →
`/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia`, or
[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (MTP launchers, or tighter memory) →
`/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat`. These are the paths the launchers check
(`MODEL_HOST_PATH`); override with `MODEL_HOST_PATH=…` if you keep weights elsewhere. For
DFlash2, also fetch the drafter (2.2 GB) → `/var/tmp/models/GLM-5.3-Flash-DFlash2`.

**3. Install the SM121 top-k fix** on **both** nodes. Both published images still contain a
decode-time top-k kernel that hard-kills the engine on any decode past ~24K context on GB10
(the launcher bind-mounts the fix over it):

```bash
mkdir -p ~/patches
cp docker/sparse_attn_indexer_kpool_sm121.py ~/patches/sparse_attn_indexer_kpool.py
```

Why, and what the crash looks like: [docs/SM121-CRASH-FORENSICS](docs/SM121-CRASH-FORENSICS-2026-08-27.md).

**4. Edit the launcher for your fabric.** Set the IPs/paths at the top of
[`launch-glm53-vllm-tp2-dflash2.sh`](launch-glm53-vllm-tp2-dflash2.sh), **and the NCCL
interface names inside the `docker run` body** (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`,
`NCCL_IB_ADDR_RANGE`) — wrong NIC names fail silently. `ibdev2netdev` and `ip -br a` will
tell you what yours are.

Then launch — **pre-launch ritual on BOTH nodes, every time** (GB10 unified memory):

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches      # both nodes
./launch-glm53-vllm-tp2-dflash2.sh 1    # worker FIRST
sleep 25
./launch-glm53-vllm-tp2-dflash2.sh 0    # then head — serves :8000
```

Without the drafter, use [`launch-glm53-vllm-tp2.sh`](launch-glm53-vllm-tp2.sh) (v8 image,
MTP-4, ~21.8 tok/s) — same ritual.

**5. Smoke test.** Wait for readiness first (~15 min; the shard load dominates). Poll
`/health`, **never `/v1/models`** — that returns 200 even with a dead engine:

```bash
until curl -sf http://<head>:8000/health >/dev/null; do sleep 20; done
```

```bash
curl http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"2+2=?"}],
       "max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}'
```

Full serve args, NCCL fabric env, and the rationale for every flag:
[docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md).

<details>
<summary><b>Build the images yourself instead</b> (only if you want to modify the patches)</summary>

The base is the public day-0 image:

```bash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130     # or -x86_64- for x86
cd docker
for i in $(seq 1 9); do
  docker build -f "Dockerfile.glm53-sm121-v$i" -t "ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v$i" . || break
done
docker save ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8 | ssh <worker> docker load
```

The chain is mostly linear (v1→v3→v4→…→v9); **v2 is an optional NaN-debug branch off v1** that nothing else builds on. For DFlash2, build
[`overlay-dflash2/`](overlay-dflash2/) on top of v8 afterwards.

> **Resolved.** The `FROM` lines of v2–v6 once referenced the original ad-hoc tag names
> (`sm121-nope-mla`, `sm121-fi618`, `sm121-fi618-nccl`, `sm121-final`) rather than
> `sm121-v1…v5`, so the loop above did not work. [#5](../../pull/5) (thanks @ozskywalker)
> normalized the chain to v-numbered tags and added `docker/build.sh`; no manual retagging
> between stages is needed.

Published image digests, so you can tell whether a local build differs:
`sm121-v8` → `sha256:d77d375c742fc54f436dec5108b440f58f021bc6600052bf0e8fe5840357e78f` ·
`sm121-v11-dflash2` → `sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`

Both tags predate `overlay-dflash2/patch_prefix_cache_draft_group.py`, so a pull of
`sm121-v11-dflash2` still has the zero-hit coordinator; rebuild the overlay for the fix.
</details>

---

## Results (TP2, 2026-08-28)

**DFlash2 + fp8 KV — the shipped configuration.** All measured on our own hardware; the
harness is [`probes/bench_c1c6.py`](probes/bench_c1c6.py), run as:
`python3 probes/bench_c1c6.py --url http://<head>:8000 --rounds 2 --max-tokens 400`

| | value |
|---|---|
| Single-stream decode, code prompt, warm | **46.9 tok/s** at 74.1 % draft acceptance |
| Single-stream decode, structured output | **54–61 tok/s** (temp 0, 3 runs) |
| KV pool | **678,661 tokens** @ 262K context at the shipped 6 GiB pin (PR #16; 581,040 under profiler sizing before it, see the ceiling section) |
| Context | 262,144 |
| KV cost of the drafter | **zero** — it slot-shares the MLA tensors |
| Boot | ~15 min (shard load dominates) |

Concurrency sweep — 2 waves/level, 400-token generations, mixed code+prose, **zero failures**:

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| aggregate tok/s | 35.1 | 41.6 | 40.6 | 47.5 | **56.2** | 47.7 |
| per-stream tok/s | 35.1 | 23.2 | 17.3 | 15.3 | 17.5 | 13.3 |
| accepted ÷ drafted | 0.53 | 0.45 | 0.42 | 0.40 | 0.51 | 0.40 |

**The progression**, same fleet, same prompt style:

| config | decode | note |
|---|---|---|
| bf16, no speculation | 14.3 tok/s | |
| fp8 KV + MTP-4 | 21.8 tok/s | previous flagship |
| **fp8 KV + DFlash2** | **46.9 tok/s** | **2.15x** — this repo |
| fp8 KV + DFlash2 at TP4 | 68.5 tok/s | four nodes, 1M context — [sibling repo](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) |

Throughput tracks how *predictable* the output is, not just the config: structured/list/
tool-argument output drafts at ~0.9 acceptance, freeform prose nearer 0.33. Agentic traffic
lives in the high-acceptance zone. Detail and how to read these:
[docs/BENCH-C1-C6-DFLASH2.md](docs/BENCH-C1-C6-DFLASH2.md).

### KV pool ceiling on TP2 (2026-08-28)

> **Superseded 2026-09-02, the shipped launcher now pins `--kv-cache-memory 6442450944`
> (6 GiB).** The guidance below ("let the profiler size the pool, never pin") was written
> before we measured what the profiler-sized pool costs under load: at the 3 GiB pin
> @tmooch measured 6 preemptions under load and 0 at 6 GiB, with the pool going
> **310,292 → 678,661 tokens**. Preemption under load costs more than any tok/s figure and
> the memory is available, so 6 GiB was adopted as the default in
> [#16](../../pull/16). Measurement and reasoning:
> [docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md](docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md).
> The sharp edge below is still real, a pin removes the activation reservation, which is
> why the shipped pin is a *measured* one, validated under load, not a guess. Do not raise
> it without repeating that measurement.

The original 2026-08-28 finding, kept because the mechanism still matters:

**Let vLLM's profiler size the pool. Do not pin `--kv-cache-memory`.**

That was the lesson at the time, and it cost us a night of boots to learn. When you pass
`--kv-cache-memory`, vLLM still runs the profile pass but **never subtracts the measured
activation peak** (`gpu_worker.py:475-495`) — it hands you exactly the number you asked
for and `--gpu-memory-utilization` becomes dead. Allocation succeeds, warmup succeeds, a
short generation succeeds, and then the first long prompt has nowhere to put its
activations and the engine dies. We reproduced that failure at four different pins.

Profiler-sized figures, all at `--max-model-len 262144` so they are comparable:

| config | KV pool | verified |
|---|---|---|
| **DFlash2 + fp8 KV** | **581,040 tokens** | serving; survived a 28,818-token prompt with the engine healthy after |
| **No drafter, fp8 KV** | **965,166 tokens** | allocated and booted; long-prompt survival not yet confirmed |

**The DFlash2 drafter costs ~4.8 GiB of KV headroom** — far more than its 2.2 GiB of
weights — which is a real trade nobody had priced: roughly **+91 % decode speed for −40 %
pool**. Choose per workload.

Three traps worth knowing before you tune this yourself:

- **The reported pool inflates with context.** `GPU KV cache size` is
  `int(max_concurrency × max_model_len)` (`kv_cache_utils.py:2264`), so raising
  `--max-model-len` raises the headline number without adding a byte of memory. Only
  `blocks × block_size`, or bytes/token, compares honestly across configs — and it is why
  pool figures published at 900K or 1M context are not comparable to figures at 262K.
- **`Available KV cache memory` is logged by rank 0 only, but the pool is built from the
  minimum across ranks** (`kv_cache_utils.py:2554`). One of our boots logged 6.25 GiB and
  bound 2.29 GiB. Read that line on **every** rank before trusting it.
- **The TP worker rank profiles 4–5 GiB less KV headroom than the head**, reproducibly, on
  both of our node pairs, independent of the drafter, of NFS, and of which physical machine
  is which. That asymmetry caps the pool and we have not explained it; it looks like an
  upstream vLLM question rather than a configuration error.

Operational note: on these 121 GiB unified-memory nodes, `vm.swappiness=0` is mandatory
and **does not survive a reboot**. With swap active the kernel pages vLLM out mid-load and
triggers a UVM driver livelock — one worker thread spinning at 100 %+ CPU, `UVM GPU` kthread
hot, GPU reporting high utilization at idle wattage, shard loading frozen at a reproducible
point. It does not recover on its own.

> A previous revision of this section published a 727,583-token figure at a 7 GiB pin as
> the usable ceiling. That configuration is **unsafe** (pinned, so no activation headroom),
> and no log of that measurement survived — it was quoted from a terminal session rather
> than a captured file. It has been withdrawn rather than restated.

### Tuning notes

- **`temperature: 0` is free throughput** (+13–21 %). vLLM's rejection sampler does an exact
  top-1 match at temp 0 but a probabilistic ratio test above it — and since the draft method is
  greedy, the draft probability is pinned to 1, making the T>0 test strictly harder.
- **`enable_thinking: false` is also the faster setting** (+8 % acceptance) — reasoning traces
  are higher-entropy and draft worse. Caveat: with thinking off GLM emits untagged
  reasoning-prose into `content`, which some agent harnesses mis-parse; see the deploy report.
- **K=7 is the default, but it is a workload choice, it was swept.** Conditional per-position
  acceptance is nearly flat (0.93/0.89/0.84/0.81/0.79/0.59/0.94), so the last position still
  earns; the drafter's `block_size: 8` caps K at 7 anyway, and a lower K gives a prettier
  *ratio* and worse throughput on a single stream. **But the sweep on 2026-09-02
  ([docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md](docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md),
  "Speculative depth: k=7 below C4, k=5 above it") found it inverts under concurrency**, 
  k=5 beats k=7 by +29.9% at C4 and +18.3% at C6, while losing on every single-stream prompt.
  That doc's result, quoted: "`num_speculative_tokens` is a **workload choice, not a
  default.** It inverts at C4… **The default stays k=7** (single-user and low-concurrency is
  the common case, and it is what the README's headline figures use). Serving deep
  concurrency should set k=5, or better, use a schedule with the crossover at C4:
  `"num_speculative_tokens_per_batch_size": [[1,3,7],[4,512,5]]`"
  A second independent pair (#21, 2026-09-16) re-measured after the prefix-cache repair and found the
  acceptance regime had moved (per-position ~100/97/96/91/90%, mean 4.75/5): k=7 beat k=5 by +19% on
  structured single-stream, was flat on prose, and lost 9% at C6. Same conclusion from the other
  direction: k is a workload choice, and the crossover sits around C4-C6.
- **Prefix cache now hits with DFlash2** once the overlay includes `patch_prefix_cache_draft_group.py` (#13). Agent sessions stop re-prefilling the whole conversation each turn: repeated 262K prompt, 0.986 hit rate, warm TTFT 3 s vs 192 s cold. See [PREFIX-CACHE-DFLASH2-SM121](docs/PREFIX-CACHE-DFLASH2-SM121.md).

---

- **Multi-user long context: keep `--max-num-seqs 1` until #14 is closed.** Two concurrent
  ~114K-token requests collapse decode to ~2 tok/s each on some TP2 pairs (#14, reproduced on
  a 4 GiB / 414K-token KV pin in #19, so it is not KV-pool exhaustion). The leading
  explanation is the DSA indexer: no varlen path off SM100, so each draft token costs one
  full-context indexer row per sequence. A lower k shrinks exactly that cost, which is the
  other reason long-context agent traffic wants k below 7. Single-user long context and
  multi-user short prompts are unaffected.
- **Pin the drafter revision.** `incoai/GLM-5.3-Flash-DFlash2` has shipped different bytes
  under the same tag (#7: `model.safetensors` sha256 `b33c0347...` vs `8931dc52...`, pulled
  days apart). Two pairs on the same recipe measured 0.73 acceptance and one measured 0.35,
  with that hash mismatch sitting right there. If acceptance is far below the numbers in
  [Results](#results-tp2-2026-08-28), re-pull the drafter by explicit revision before
  touching anything in the image chain, and run the count-to-100 probe at temperature 0
  (`Count from 1 to 100. Output only the numbers, one per line, nothing else.`): a healthy
  engine returns ~0.9 acceptance on it regardless of workload.

## Status: work in progress

This repo is an active bring-up log, not a finished product. Everything published here is
measured on our own hardware and dated — **if a number is not in a dated table, treat it as
unverified.**

| | state |
|---|---|
| **DFlash2 + fp8 KV, TP2** | ✅ **Proven.** Benchmarked C1–C6, zero failures. This is the config to copy. |
| **DFlash2 + fp8 KV, TP4 / 1M ctx** | ✅ Serving — 68.5 tok/s, 2,622,494-token pool. See the sibling repo. |
| **DFlash2 + NVFP4 KV, TP2** | ⚠️ **Partial.** Serves and drafts (35.9 tok/s, 0.563 acceptance, 334K pool), but prompts long enough to need chunked prefill (>~3K tokens) kill the rank-0 worker. Root cause open; the standalone drafter KV path is the suspect. [Details](docs/DFLASH2-SPECULATIVE-DECODING.md). |
| **InstantTensor fast load** | ⚠️ Experimental — 15x faster loads, unstable multi-node. See below. |

---

## Why this needs a patched image

The vLLM PR authors' day-0 image (`vllm/vllm-openai:glm53-flash-arm64-cu130`) works on B200.
On GB10/SM121 it fails five separate ways. Our derivative (`docker/Dockerfile.glm53-sm121*`,
applied in order) fixes those five, then adds a sixth patch that unlocks fp8 KV:

1. **NoPE MLA vs the SM12x sparse backend** — the only stock capability-12 sparse-attention
   backend requires the packed `fp8_ds_mla` layout, which hardcodes DeepSeek's `pe_dim=64`.
   GLM-5.3 is NoPE (`qk_rope_head_dim=0`) → assert death in warmup. Fix: extend vLLM's SM90
   NoPE sparse-MLA backend to SM121 with the FA2 path — probed on-GPU with the model's real
   shape before trusting it (`probes/probe_sm121_nope_mla.py`).
2. **FlashInfer 0.6.17 FA2 MLA NaN** — the FA2 scheduler produces NaN for 64–256-row batches
   on SM121 (bisect: `probes/probe_fa2_bisect.py`), and normal prompts land exactly there.
   Fix: FlashInfer **0.6.18 nightly**.
3. **The nightly's dependency sabotage** — it silently downgrades `nvidia-nccl-cu13` to 2.29.7
   (NCCL "internal error" on the Spark IB fabric; re-pin **2.30.7**) and skews
   `nvidia-cutlass-dsl` to a mixed 4.7.0/4.6.2 state (CuTeDSL warmup ICE; re-pin **4.6.2**).
   Audit transitive pins after ANY pip install in these images.
4. **PDL on unvalidated silicon** — vLLM enables Programmatic Dependent Launch for capability
   ≥ 9, including SM121, in the Triton kernels carrying KDA recurrent state. Gated off on SM12x.
5. **Indexer uninitialized top-k** — the kpool top-k destination was `torch.empty` and the
   kernels only guarantee the first `min(k, valid)` entries; short rows carried garbage pool
   ids → bogus token indices → NaN lottery. Fix: init to `-1` + clamp (`docker/patch_v7.py`).
6. **fp8 KV cache unlock** (v8, `docker/patch_v8_fp8.py`) — see below.

Two serve-flag landmines, no code needed:

- **`--block-size 2304`** — vLLM's hybrid block aligner picks a size whose kpool storage tiles
  by 32, but DeepGEMM's arch-12 fp8 paged-MQA accepts only 64-entry pool pages. 2304 is a
  multiple of kpool·64 and of the MLA 128 alignment.
- **`--gpu-memory-utilization 0.85`** — 0.78–0.80 starve the KV cache at 131K+.
  (Credit: barrydeen's independent recipe.)

### fp8 KV cache on GB10: a two-line fix (and, as far as we can tell, a first)

FlashInfer gates fp8 MLA KV to SM90, and naively relaxing the gate fails with CUDA "invalid
argument" (`probes/probe_fa2_fp8.py`). The real cause (`docker/patch_v8_fp8.py`): the fa2 fp8
branch **forces `CTA_TILE_KV=32`, a Hopper 228KB-smem assumption**. On GB10's ~101KB opt-in max
that over-requests shared memory (117,312 B > 101,376 B) at `cudaFuncSetAttribute`, before the
kernel ever launches. **Capping the tile instead of forcing it** (fp8 keeps TKV=16 on
100KB-class devices — 91,680 B, fits) makes fp8 KV work: verified on-GPU, rel-err ~0.005 vs an
fp32 reference, then end-to-end in production.

As far as we can tell this is the first fp8 KV cache for a NoPE-MLA model on any consumer
Blackwell part. Upstream-ready issue drafts with receipts:
`docs/issue-flashinfer-fp8-mla-sm121.md`, `docs/issue-vllm-nope-fp8-ds-mla.md`.

---

## Hard-won operational rules

- **Tear down BOTH ranks before relaunching either.** A rank that rendezvouses with a dying one
  hangs or dies confusingly.
- **`grep '^IMAGE' launch-*.sh` on BOTH nodes before every launch.** Two "mystery" garbage boots
  were a silent image-version mismatch between ranks. Copy whole files between nodes; never
  `sed` over ssh.
- **Capture `docker logs` before `docker rm -f`.**
- **Probe `/health` for liveness, never `/v1/models`** — the latter returns 200 from config alone, with a dead engine behind it.
- **Two consecutive unexplained deaths = stop and diagnose.** Never crash-loop.
- **Swap on with `vm.swappiness=0`** — not off. Fully disabled, the worker dies during MoE
  marlin repack with no valve; at default swappiness the kernel pages vLLM out mid-load and
  triggers a UVM driver livelock that freezes the shard loader.
- `max_tokens` includes reasoning tokens when thinking is on; disable per-request with
  `chat_template_kwargs: {"enable_thinking": false}`.

---

- **Do not benchmark on a fresh boot; warm the batch shapes first (#21).** The first
  concurrent burst after a cold start can trigger a TileLang JIT compile mid-batch
  (`mhc_pre_big_fuse_with_norm_tilelang` in the log at the exact moment of the burst); the
  worker RPC stalls cross-rank and the engine dies with `TimeoutError: RPC call to
  sample_tokens timed out` then `EngineDeadError`. Send a few small concurrent requests
  (4 x 32-token generations) before any real load. `fleet_watchdog.sh` recovers the pair, but
  the bench you were running is gone.

## Deeper reading

| doc | what's in it |
|---|---|
| [DEPLOY-REPORT](docs/DEPLOY-REPORT.md) | the seven day-0 bugs, root causes, receipts, every serve flag |
| [DFLASH2-SPECULATIVE-DECODING](docs/DFLASH2-SPECULATIVE-DECODING.md) | the drafter port: four patches, the KV-layout fix, nine boots of failure modes |
| [PREFIX-CACHE-DFLASH2-SM121](docs/PREFIX-CACHE-DFLASH2-SM121.md) | why the prefix cache never hit with the drafter group, the two-edit coordinator fix, and a 0.986 hit rate at 262K |
| [BENCH-C1-C6-DFLASH2](docs/BENCH-C1-C6-DFLASH2.md) | full concurrency tables and how to read them |
| [SM121-CRASH-FORENSICS](docs/SM121-CRASH-FORENSICS-2026-08-27.md) | why the fleet "randomly" died: a topk kernel bug and phantom KV backing |
| [GB10-KV-MEMORY-LADDER](docs/GB10-KV-MEMORY-LADDER.md) | why KV budgets above vLLM's suggestion die, and the driver-level mechanism |
| [KV-HUNT-672K-TP2-RECORD](docs/KV-HUNT-672K-TP2-RECORD.md) | the 8-attempt hunt past the 507K wall |
| **[OPEN-PROBLEMS](docs/OPEN-PROBLEMS.md)** | **everything we broke and could not fix — reproducible, with next probes. Start here if you want to contribute.** |
| [GB10-UNIFIED-MEMORY-FIELD-REPORT](docs/GB10-UNIFIED-MEMORY-FIELD-REPORT.md) | field report from a second fleet: the memory guard that ends the load-time driver failures, the min_free_kbytes trap, zero-fault warm-up, and hard power-offs with the GPU idle |

**Debugging kit** (reusable for any day-0 model on new silicon): `probes/probe_sm121_nope_mla.py`
(probe a kernel with your real geometry before patching arch gates) · `probes/probe_fa2_bisect.py`
(NaN bisect over batch shapes) · `probes/probe_mhc.py` (A/B a Triton kernel vs its torch
reference) · `probes/gb10_alloc_probe.py` (map the allocation wall) · `probes/bench_c1c6.py`.
The deploy report also describes the env-gated forward-hook NaN localizer (`GLM53_NAN_DEBUG=1`)
that names the first module emitting non-finite values.

### Fast loading: InstantTensor (experimental)

The v9 image adds the InstantTensor direct-I/O loader (`--load-format instanttensor`): loads
drop from ~10 minutes to 40–100 seconds and the page cache stays empty. **But in all four of
our v9 TP2 boots a rank died silently ~1 minute after loading** (exit code None, nothing in
dmesg) at every KV budget — so the shipped launchers do not enable it and the stable image
remains v8. This matches the known multi-node instability class for direct-IO loaders on Spark
(eugr/spark-vllm-docker#29). Because direct I/O never fills the page cache it also defeats the
first layer of the GB10 KV-allocation wall — full story in
[docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md). Credit: jack6464 (NVIDIA forum).

### vLLM v0.28.0 status (checked 2026-08-27)

**Not viable for GLM-5.3 yet**: the `glm5_next` architecture is not in the v0.28.0 release
(vllm-project/vllm#53906 still unmerged at check time) and no rebased day-0 image exists. The
day-0 image used here is itself a main-branch dev snapshot (`0.1.dev20051`) cut near the 0.28
branch point — this stack already runs 0.28-era engine code *plus* the GLM support 0.28 lacks.
Porting is mechanical when it opens: the patches are guarded string-replacements that apply or
refuse loudly.

---

## Credits

- **New default serving stack (2026-09-29)**: [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) by [@knapcio](https://github.com/knapcio), ported here to TP2
- **Model**: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) ·
  **Quant**: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (DFlash2 default) and [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (MTP, compressed-tensors)
  (their sm_121 notes were used directly) ·
  **Drafter**: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
- **barrydeen** — the gmu 0.85 reference config and quantization-coverage table from their
  independently published DGX Spark recipe
- **@ozskywalker** — [#5](../../pull/5), Dockerfile tag chain + build script
- [Matt Mastracci](https://github.com/mmastrac): the [GLM-5.3-Flash GX10 recipe](https://github.com/kindlingai/glm-5.3-flash-gx10) behind the KDA conv split, sparse-MLA prefill and MoE prefill ideas in knapcio's stack, the FlashKDA fp32-state kernels, and vLLM [PR #58454](https://github.com/vllm-project/vllm/pull/58454)
- vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 image;
  FlashInfer for the 0.6.18 SM90-NoPE MLA path; upstream
  [PR #52816](https://github.com/vllm-project/vllm/pull/52816) for DFlash2
- Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild)

## Experimental

- [NVFP4 attention and MLP projections](docs/EXPERIMENTAL-NVFP4-ATTENTION.md) - 18 GiB of this checkpoint class sits in bf16 and is read every decode step. Quantizing 13.88 GiB of it measured x1.14-1.30 per category at TP4. **Untested at TP2**; the arithmetic predicts about x1.21 and frees 5.21 GiB/rank, which is more than the KV bump was worth on 2026-09-18. May show a speed increase here, not yet verified.
