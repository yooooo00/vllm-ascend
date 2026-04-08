# Nemotron 3 Super migration notes

## Source baseline

Reference baseline used for the original adaptation:

- `vllm`: `d5e6d3c5f07a41df73c388f96af533ec2973ebfe`
- `vllm-ascend`: `453c54a8fa6aacbaabfc23fcabe7f8a5048ed3c1`

## Target baseline

Current maintained upstreams used for this port:

- `vllm`: `upstream/main` (`78434b923`)
- `vllm-ascend`: `upstream/main` (`dae3c99e`)

## What changed between old baseline and current main

### 1. Modular MoE contract drift

Current `vllm` main expects backend methods to expose newer modular-MoE behavior:

- `is_monolithic` is queried during layer construction
- unquantized MoE `apply(...)` now receives routed weights/ids directly
- shared-expert execution and return semantics follow the current `SharedFusedMoE` runner contract
- `unified_apply_mlp(...)` now expects unpacked arguments instead of a single `MoEMlpComputeInput`

This required the bulk of the Nemotron port work in:

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm_ascend/ops/fused_moe/moe_mlp.py`

### 2. Worker/input metadata drift

Current `vllm` main requires additional worker-side fields and metadata propagation:

- `NPUInputBatch.logprob_token_ids`
- `CommonAttentionMetadata.is_prefilling`

These were handled in:

- `vllm_ascend/worker/npu_input_batch.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/attention/utils.py`

### 3. Mamba state-passing call signature drift

The old fallback expected per-sequence chunk layout inputs, while current `vllm` main calls into state passing with `last_chunk_indices`.

This was adapted in:

- `vllm_ascend/patch/worker/patch_triton.py`

### 4. Build-time robustness

Torch/Torch-NPU emits extra text in this environment during version probing. CMake version detection was made more robust in:

- `CMakeLists.txt`

## Code paths intentionally kept in vllm-ascend

All model-specific enablement in this port stays inside `vllm-ascend`. No `vllm` source patch is required for the current working branch.

## Follow-up recommendations

1. Evaluate `fix_mistral_regex=True` for the tokenizer path and decide whether this should be surfaced in docs or service flags.
2. Expand validation beyond seq=2 after the functional port is merged.
3. Revisit performance tuning only after mainline compatibility is merged and stable.
