# NVIDIA Nemotron 3 Super validation

## Environment

- system Python: `/usr/local/python3.11.14/bin/python3`
- CANN env: `/usr/local/Ascend/cann-8.5.1/set_env.sh`
- model: `/opt/data/c00913822/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- runtime workspace: `/vllm-workspace-port`

## Service launch template

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export PYTHONPATH=/vllm-workspace-port/vllm:/vllm-workspace-port/vllm-ascend:${PYTHONPATH:-}
export VLLM_PLUGINS=ascend
export TRITON_ALL_BLOCKS_PARALLEL=1
export OMP_NUM_THREADS=128
unset HCCL_OP_EXPANSION_MODE

/usr/local/python3.11.14/bin/python3 -B -m vllm.entrypoints.cli.main serve \
  /opt/data/c00913822/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --host 127.0.0.1 \
  --port 18032 \
  --served-model-name nemotron-super \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
```

## Validation matrix

| Mode | max-num-seqs | Endpoint(s) | Result | Evidence |
|---|---:|---|---|---|
| eager | 1 | `/v1/completions`, `/v1/chat/completions` | pass | `/opt/data/c00913822/nemotron_port_20260408/validation/eager_seq1_retry26/` |
| eager | 2 | `/v1/completions`, `/v1/chat/completions` | pass | `/opt/data/c00913822/nemotron_port_20260408/validation/eager_seq2_retry1/` |
| graph (`FULL_DECODE_ONLY`, `[1,2]`) | 1 | `/v1/completions`, `/v1/chat/completions` | pass | `/opt/data/c00913822/nemotron_port_20260408/validation/graph_seq1_retry1/` |
| graph (`FULL_DECODE_ONLY`, `[1,2]`) | 2 | `/v1/completions`, `/v1/chat/completions` | pass | `/opt/data/c00913822/nemotron_port_20260408/validation/graph_seq2_retry1/` |
| graph (`FULL_DECODE_ONLY`, `[1,2]`) | 2 | small benchmark | pass | `/opt/data/c00913822/nemotron_port_20260408/validation/graph_seq2_perf2/benchmark_cli.log` |

## Output samples

Graph seq=2 post-benchmark smoke:

- completions: `Do not add any extra text or explanation." So the answer must be exactly "`
- chat: `User wants exactly "zebra-42" and nothing else. So`

These outputs are coherent and non-garbled. They do not strictly obey the exact-output instruction, but the same behavior is observed consistently across eager/graph smoke tests.

Artifacts:

- `/opt/data/c00913822/nemotron_port_20260408/validation/graph_seq2_perf2/completions_after_bench.json`
- `/opt/data/c00913822/nemotron_port_20260408/validation/graph_seq2_perf2/chat_after_bench.json`

## Targeted tests

Executed with system Python:

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export PYTHONPATH=/vllm-workspace-port/vllm:/vllm-workspace-port/vllm-ascend:${PYTHONPATH:-}
cd /vllm-workspace-port/vllm-ascend
/usr/local/python3.11.14/bin/python3 -m unittest \
  tests.ut.ops.test_fused_moe_nemotron \
  tests.ut.ops.test_causal_conv1d_nemotron
```

Result:

- `Ran 7 tests ... OK`
- log: `/opt/data/c00913822/nemotron_port_20260408/build_logs/unittest_local_20260408.log`

## Small graph benchmark

Command:

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export PYTHONPATH=/vllm-workspace-port/vllm:/vllm-workspace-port/vllm-ascend:${PYTHONPATH:-}
/usr/local/python3.11.14/bin/python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 18032 \
  --endpoint /v1/chat/completions \
  --model nemotron-super \
  --tokenizer /opt/data/c00913822/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --dataset-name random \
  --num-prompts 4 \
  --max-concurrency 2 \
  --random-input-len 256 \
  --random-output-len 64 \
  --ignore-eos \
  --trust-remote-code
```

Result:

- output throughput: `18.98 tok/s`
- total throughput: `94.91 tok/s`
- mean TTFT: `4788.07 ms`
- mean TPOT: `30.85 ms`
- benchmark log: `/opt/data/c00913822/nemotron_port_20260408/validation/graph_seq2_perf2/benchmark_cli.log`
