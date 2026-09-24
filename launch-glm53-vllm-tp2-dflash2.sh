#!/usr/bin/env bash
#
# GLM-5.3-Flash + DFlash2 speculative decoding, TP2 on 2x DGX Spark (GB10/SM121).
# This is the configuration behind the README's 46.9 tok/s figure.
#
# Prerequisites (see README Quickstart):
#   1. docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2  (public image; pull on BOTH nodes)
#   2. weights at $MODEL_HOST_PATH on BOTH nodes
#   3. drafter (2.2 GB) at /var/tmp/models/GLM-5.3-Flash-DFlash2 on BOTH nodes
#   4. cp docker/sparse_attn_indexer_kpool_sm121.py $HOME/patches/sparse_attn_indexer_kpool.py
#      on BOTH nodes -- the SM121 top-k fix. Without it the engine dies on any
#      decode past ~24K context (docs/SM121-CRASH-FORENSICS-2026-08-27.md).
#
# Usage: ./launch-glm53-vllm-tp2-dflash2.sh <0|1>   -- worker (1) FIRST, then head (0)
#
# 2026-09-18 speed night (docs/SPEED-NIGHT-2026-09-18.md): boot hardening (JIT storm cap,
# persistent kernel caches, cgroup cap, video profile off), KV pin 6 -> 8 GiB, the #18
# prefix-cache repair and the b12x RoCE all-reduce are on by default when their files are
# present. ROCE=0 / PREFIX_FIX=0 turn them off. Build the RoCE bundle once per node with
# speed-night-2026-09-18/roce/build-roce-bundle.sh; the prefix file with
# docker/dflash2-overlay/patch_prefix_cache_draft_group.py (see the doc).
set -euo pipefail

# GLM-5.3-Flash-NVFP4 on Reddie (head, rank 0) + Spark4 (worker, rank 1), vLLM TP2 over the fabric.
# Official day-0 image (vLLM has glm5_next; SGLang support for this NVFP4 quant is still in-flight).
# Day-1: NO speculative decode. MTP phase-2 after base is stable (image has Glm5NextMTPModel).
# Run worker FIRST: Spark4 rank 1, wait ~20s, then Reddie rank 0.
NODE_RANK="${1:?usage: launch-glm53-vllm-tp2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2"
NAME="vllm_glm53"
# Checkpoint. Partial-attention ModelOpt builds (LibertAIDAI, abliterated variants) emit
# intermittent corrupted token IDs -- vLLM #54150, 4/9/8 U+FFFD on a Hangul probe. The nvidia
# build keeps every layer's attention in high precision and is clean (0/0/0, issue #23), as is
# RedHatAI (compressed-tensors). Override with MODEL_HOST_PATH=... for other builds.
# Default checkpoint: nvidia/GLM-5.3-Flash-NVFP4 (issue #23; clean on the Hangul probe, attention
# kept in high precision). Falls back to the RedHatAI copy if only that one is on disk.
if [ -z "${MODEL_HOST_PATH:-}" ]; then
  MODEL_HOST_PATH=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia
  [ -f "$MODEL_HOST_PATH/config.json" ] || MODEL_HOST_PATH=/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat
fi
MODEL_PATH="/models/glm-5.3-flash-nvfp4"

# Guard (tools/checkpoint_guard.py): refuse ModelOpt builds that quantize attention (the
# corrupting shape) unless ALLOW_MODELOPT=1. The corruption is nearly invisible in English
# prose and only bites inside tool-call blocks, so it will not announce itself at boot.
python3 "$(dirname "$0")/tools/checkpoint_guard.py" "$MODEL_HOST_PATH" dflash2 || exit 5

CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.192.2"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS="" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless" ;;
esac

test -f "$MODEL_HOST_PATH/config.json"
mkdir -p "$CACHE_HOST_PATH/flashinfer" "$CACHE_HOST_PATH/tilelang" "$CACHE_HOST_PATH/triton"
docker rm -f "$NAME" 2>/dev/null || true

# #18 prefix-cache repair for the drafter group (kv_cache_coordinator.py patched inside the
# v11 image and bind-mounted). Measured 2026-09-18: 13K-token repeat TTFT 21.1 s -> 6.3 s.
PREFIX_MOUNT=""
if [ "${PREFIX_FIX:-1}" = "1" ] && [ -f "$HOME/patches/kv_cache_coordinator.py" ]; then
  PREFIX_MOUNT="-v $HOME/patches/kv_cache_coordinator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py:ro"
elif [ "${PREFIX_FIX:-1}" = "1" ]; then
  echo "note: ~/patches/kv_cache_coordinator.py missing -> prefix-cache repair (#18) OFF" >&2
fi

# b12x RoCEnante one-shot RoCE all-reduce (local-inference-lab/vllm#597 ported to this tree).
# Measured 2026-09-18: aggregate +5-18% at C1-C6 on a compute-clamped fleet, no regressions.
ROCE_MOUNTS=""; ROCE_ENV=""
if [ "${ROCE:-1}" = "1" ] && [ -d "$HOME/patches/glm-roce/b12x" ]; then
  R="$HOME/patches/glm-roce"; D=/usr/local/lib/python3.12/dist-packages
  ROCE_MOUNTS="-v $R/b12x:$D/b12x:ro -v $R/b12x-1.3.0.dist-info:$D/b12x-1.3.0.dist-info:ro -v $R/b12x-roce:/opt/b12x-roce:ro
    -v $R/b12x_roce_all_reduce.py:$D/vllm/distributed/device_communicators/b12x_roce_all_reduce.py:ro
    -v $R/cuda_communicator.py:$D/vllm/distributed/device_communicators/cuda_communicator.py:ro
    -v $R/parallel_state.py:$D/vllm/distributed/parallel_state.py:ro
    -v $R/envs.py:$D/vllm/envs.py:ro
    -v $R/gpu_worker.py:$D/vllm/v1/worker/gpu_worker.py:ro"
  ROCE_ENV="-e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB -e VLLM_ROCE_ALLGATHER_ENABLE=1 -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache"
elif [ "${ROCE:-1}" = "1" ]; then
  echo "note: ~/patches/glm-roce missing -> RoCE all-reduce OFF (NCCL). Build it with speed-night-2026-09-18/roce/build-roce-bundle.sh" >&2
fi

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  `# cgroup cap: an overrun is a clean container OOM, not the host page-allocator livelock` \
  `# + watchdog reboot that took three nodes down on 2026-09-18 (UMA GPU allocs are not` \
  `# cgroup-charged; this bounds the host side only)` \
  --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  `# FlashInfer JIT output (fp4 CUTLASS GEMM variants, minutes each) lives in` \
  `# /root/.cache/flashinfer and dies with the container unless persisted` \
  -v "$CACHE_HOST_PATH/flashinfer:/root/.cache/flashinfer" \
  $PREFIX_MOUNT $ROCE_MOUNTS \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  `# JIT storm control: with default MAX_JOBS, cicc (nvcc) invoked the OOM killer during the` \
  `# post-load profile on 2026-09-18. Keep TileLang/Triton caches across boots.` \
  -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  $ROCE_ENV \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
  -v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-seqs 6 --block-size 2304 --moe-backend marlin --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}' --kv-cache-dtype fp8_e4m3 --kv-cache-memory 8589934592 \
    --enforce-eager --max-num-batched-tokens 8192 \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking":false}' --chat-template /models/glm-5.3-flash-nvfp4/chat_template_mm.jinja \
    `# images stay on; video=0 skips the max-size video encoder profile at boot (memory spike)` \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
