# NVIDIA Nemotron 3 Super

## Introduction

NVIDIA Nemotron 3 Super is a large Mixture-of-Experts language model that uses
the `NemotronH` architecture together with Mamba and shared-expert execution
paths. This document summarizes the validated vLLM Ascend bring-up path for
[`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
on current `vllm-ascend` main.

This port focuses on functional compatibility on Ascend. The validation in this
document covers:

- BF16 serving on a single 8-card Ascend A2 / 910B system
- `/v1/completions` and `/v1/chat/completions`
- eager mode with `max-num-seqs=1/2`
- `FULL_DECODE_ONLY` graph mode with capture sizes `[1, 2]`

It does not claim long-context qualification, higher-concurrency tuning, or
peak-throughput optimization.

## Supported Features

Refer to [supported models](../../user_guide/support_matrix/supported_models.md)
for the model support matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) for feature
configuration details.

Validated scope for this port:

| Item | Status | Notes |
|------|--------|-------|
| BF16 serving | validated | single-node TP8 on Ascend A2 / 910B |
| `/v1/completions` | validated | eager and graph, `max-num-seqs=1/2` |
| `/v1/chat/completions` | validated | eager and graph, `max-num-seqs=1/2` |
| `FULL_DECODE_ONLY` graph mode | validated | `cudagraph_capture_sizes=[1,2]` |
| Expert parallel | not validated | follow-up work |
| Prefix cache | not validated | follow-up work |
| Context longer than 4096 | not validated | this port only qualifies `--max-model-len 4096` |

## Environment Preparation

### Model Weight

- `NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (BF16 version):
  [Download model weight](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)

Validated hardware for this document:

- 1 Atlas A2 node (8 NPUs / Ascend 910B), `tensor-parallel-size=8`

The model weight should be downloaded to a local or shared path that is visible
inside the container, for example `/root/.cache/models/` or `/models/`.

### Installation

You can use the official vLLM Ascend docker image.

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 series
:sync: A2

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
::::{tab-item} A3 series
:sync: A3

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
:::::

Only the A2 configuration was validated for this port. Use the A3 image only if
your environment and model sharding strategy have already been qualified.

## Deployment

### Single-node Deployment

The following command is the validated graph-mode launch template from the NPU
environment. Replace `/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` with your
actual local model path.

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export PYTHONPATH=/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:${PYTHONPATH:-}
export VLLM_PLUGINS=ascend
export TRITON_ALL_BLOCKS_PARALLEL=1
export OMP_NUM_THREADS=128
unset HCCL_OP_EXPANSION_MODE

vllm serve /models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name nemotron-super \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
```

For eager-mode isolation, add:

```bash
--enforce-eager
```

## Functional Verification

Once the server is ready, verify the model list endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Completion smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-super",
    "prompt": "Return exactly this string and nothing else: zebra-42",
    "temperature": 0.0,
    "max_tokens": 32
  }'
```

Chat smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-super",
    "messages": [
      {"role": "user", "content": "Return exactly this string and nothing else: zebra-42"}
    ],
    "temperature": 0.0,
    "max_tokens": 32
  }'
```

## Accuracy Evaluation

Dedicated accuracy evaluation is not included in this functional porting PR.
Before claiming production readiness, run an evaluation workflow such as
[AISBench](../../developer_guide/evaluation/using_ais_bench.md) or
[lm_eval](../../developer_guide/evaluation/using_lm_eval.md) on the final
merged branch and record the benchmark configuration together with the result.

## Performance

This port includes only a small graph-mode sanity benchmark, not a full tuning
pass.

Validated benchmark command:

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/chat/completions \
  --model nemotron-super \
  --tokenizer /models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --dataset-name random \
  --num-prompts 4 \
  --max-concurrency 2 \
  --random-input-len 256 \
  --random-output-len 64 \
  --ignore-eos \
  --trust-remote-code
```

Observed result on the validated A2 setup:

| Metric | Value |
|--------|-------|
| Output throughput | `18.98 tok/s` |
| Total throughput | `94.91 tok/s` |
| Mean TTFT | `4788.07 ms` |
| Mean TPOT | `30.85 ms` |

Targeted regression tests executed in the NPU environment:

- `tests.ut.ops.test_fused_moe_nemotron`
- `tests.ut.ops.test_causal_conv1d_nemotron`
