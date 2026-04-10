# [Model][Ascend] Port NVIDIA Nemotron 3 Super support to vllm-ascend main

## Background

This branch ports a previously validated NVIDIA Nemotron 3 Super Ascend enablement from the old reference baseline

- `vllm`: `d5e6d3c5f07a41df73c388f96af533ec2973ebfe`
- `vllm-ascend`: `453c54a8fa6aacbaabfc23fcabe7f8a5048ed3c1`

onto current maintained upstream branches:

- `vllm`: `upstream/main` (`78434b923`)
- `vllm-ascend`: `upstream/main` (`dae3c99e`) plus this branch's commits

The goal is to keep the model-specific enablement inside `vllm-ascend`, preserve service correctness, and restore eager/graph serviceability for Nemotron 3 Super on Ascend.

## Why this belongs in vllm-ascend

The changes are Ascend-specific compatibility and runtime fixes:

- NPU input batch metadata expected by the Ascend worker path
- Ascend modular MoE compatibility with current `vllm` contracts
- Ascend Mamba state-passing fallback wiring for current `vllm` signatures
- Ascend attention metadata propagation for graph-safe prefilling/decode
- Ascend-specific tests for Nemotron shared experts and causal-conv decode behavior
- CMake/system build robustness for the local Torch/Torch-NPU environment

No new `vllm` core patch is required in this branch.

## Main changes

1. Update Ascend fused MoE integration to current `vllm` modular-MoE interfaces
   - add `is_monolithic` on the Ascend unquantized MoE method
   - adapt `apply(...)` to `(x, topk_weights, topk_ids, shared_experts_input)`
   - adapt `moe_comm_method._apply_mlp(...)` to the current `unified_apply_mlp(...)` signature
   - handle Nemotron's `relu2_no_mul` activation explicitly
   - seed shared-expert/gate state early enough for current layer construction order
   - preserve shared-expert return behavior expected by current `SharedFusedMoE`

2. Update worker/runtime metadata for current `vllm`
   - add `logprob_token_ids` to `NPUInputBatch`
   - compute and propagate `is_prefilling` into `AscendCommonAttentionMetadata`
   - preserve `is_prefilling` in metadata slicing/unpadding

3. Update Nemotron Mamba fallback wiring
   - port `_state_passing_fwd_npu(...)` to the current `vllm` call signature using `last_chunk_indices`

4. Add/refresh targeted regression tests
   - shared-expert init ordering
   - modular-MoE compatibility
   - `relu2_no_mul` MLP path
   - causal-conv decode regression checks

## Validation summary

Validated with system Python (`/usr/local/python3.11.14/bin/python3`) and CANN env, using model:

`/opt/data/c00913822/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`

Successful cases:

- eager, `max-num-seqs=1`
- eager, `max-num-seqs=2`
- graph (`FULL_DECODE_ONLY`, capture sizes `[1,2]`), `max-num-seqs=1`
- graph (`FULL_DECODE_ONLY`, capture sizes `[1,2]`), `max-num-seqs=2`

Service endpoints validated:

- `/v1/completions`
- `/v1/chat/completions`

Targeted unit tests:

- `tests.ut.ops.test_fused_moe_nemotron`
- `tests.ut.ops.test_causal_conv1d_nemotron`

A small graph benchmark was also captured for seq=2:

- config: graph, `max-num-seqs=2`, random `256 in / 64 out`, concurrency 2
- output throughput: `18.98 tok/s`
- total throughput: `94.91 tok/s`
- mean TTFT: `4788.07 ms`
- mean TPOT: `30.85 ms`

## Known limitations

- This PR focuses on functional porting and minimal service validation, not peak throughput tuning.
- The tokenizer still emits the upstream Mistral regex warning. Outputs are coherent and requests complete, but the fixed-regex path should be evaluated separately.
- Validation here covers `max-num-seqs=1/2`. Higher concurrency and production tuning remain follow-up work.

## Suggested PR title

`[Model][Ascend] Port NVIDIA Nemotron 3 Super support to current vllm-ascend main`
