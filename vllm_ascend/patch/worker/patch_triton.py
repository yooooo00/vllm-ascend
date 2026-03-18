import json
import os
from pathlib import Path

import torch
import vllm.model_executor.layers.fused_moe.layer
import vllm.model_executor.layers.mamba.ops.causal_conv1d
import vllm.model_executor.layers.mamba.ops.layernorm_gated
import vllm.model_executor.layers.mamba.ops.mamba_ssm
import vllm.model_executor.layers.mamba.ops.ssd_bmm
import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan
import vllm.model_executor.layers.mamba.ops.ssd_chunk_state
import vllm.model_executor.layers.mamba.ops.ssd_state_passing
import vllm.v1.worker.gpu.sample.gumbel

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_recurrent_gated_delta_rule_fwd_kernel
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu


def _device_guard(device_or_index=None):
    if isinstance(device_or_index, torch.device):
        if device_or_index.type == "npu":
            return torch.npu.device(device_or_index.index)
        return torch.cuda.device(device_or_index.index)
    if isinstance(device_or_index, int):
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu.device(device_or_index)
        return torch.cuda.device(device_or_index)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.device(torch.npu.current_device())
    return torch.cuda.device(torch.cuda.current_device())


class _CudaProxy:

    def __init__(self, cuda_mod):
        self._cuda_mod = cuda_mod

    def device(self, device_or_index=None):
        return _device_guard(device_or_index)

    def __getattr__(self, name):
        return getattr(self._cuda_mod, name)


class _TorchProxy:

    def __init__(self, torch_mod):
        self._torch_mod = torch_mod
        self.cuda = _CudaProxy(torch_mod.cuda)

    def __getattr__(self, name):
        return getattr(self._torch_mod, name)


class _PlatformProxy:

    def __init__(self, platform):
        self._platform = platform

    def is_cuda_alike(self):
        return self._platform.is_cuda_alike() or getattr(
            self._platform, "device_type", None) == "npu"

    def __getattr__(self, name):
        return getattr(self._platform, name)


def _patch_module_torch(module):
    module.torch = _TorchProxy(torch)


_DEBUG_COUNTS = {"chunk_scan": 0, "state_passing": 0}


def _debug_mamba_io(kind, payload):
    debug_path = os.getenv("VLLM_ASCEND_DEBUG_MAMBA_IO")
    if not debug_path:
        return
    min_seqlen = int(os.getenv("VLLM_ASCEND_DEBUG_MAMBA_MIN_SEQLEN", "0"))
    if payload.get("seqlen", 0) < min_seqlen:
        return
    if _DEBUG_COUNTS[kind] >= 256:
        return
    _DEBUG_COUNTS[kind] += 1
    path = Path(debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")


def _chunk_scan_fwd_npu(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    if (x.device.type != "npu"
            or os.getenv("VLLM_ASCEND_DISABLE_CHUNK_SCAN_FALLBACK") == "1"):
        return _ORIGINAL_CHUNK_SCAN_FWD(
            cb,
            x,
            dt,
            dA_cumsum,
            C,
            states,
            cu_chunk_seqlens,
            out,
            seq_idx,
            D=D,
            z=z,
            initial_states=initial_states,
        )

    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads, )
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks, )

    grid = lambda META: (
        vllm.model_executor.layers.mamba.ops.ssd_chunk_scan.triton.cdiv(
            chunk_size, META["BLOCK_SIZE_M"]) *
        vllm.model_executor.layers.mamba.ops.ssd_chunk_scan.triton.cdiv(
            headdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    _debug_mamba_io(
        "chunk_scan",
        {
            "device": x.device.type,
            "seqlen": int(seqlen),
            "nheads": int(nheads),
            "headdim": int(headdim),
            "nchunks": int(nchunks),
            "chunk_size": int(chunk_size),
            "ngroups": int(ngroups),
            "dstate": int(dstate),
            "has_initial_states": initial_states is not None,
            "seq_ids_head": seq_idx[: min(8, seq_idx.numel())].detach().cpu().tolist(),
            "cu_chunk_head": cu_chunk_seqlens[: min(9, cu_chunk_seqlens.numel())].detach().cpu().tolist(),
            "states_absmax": float(states.detach().abs().max().cpu()),
            "initial_absmax": float(initial_states.detach().abs().max().cpu()) if initial_states is not None else None,
        },
    )

    # The Triton-Ascend kernel is reliable for the single-chunk path. Multi-chunk
    # prefills compile or execute incorrectly on 910B, so use a numerically
    # equivalent torch implementation there.
    if nchunks > 1 or initial_states is not None:
        head_to_group = torch.div(
            torch.arange(nheads, device=x.device),
            nheads // ngroups,
            rounding_mode="floor",
        )
        seq_ids = seq_idx.to(torch.long)
        chunk_offsets = cu_chunk_seqlens.to(torch.long)
        chunk_lengths = chunk_offsets[1:] - chunk_offsets[:-1]
        dA_all = dA_cumsum.to(torch.float32).permute(1, 0, 2).contiguous()
        dt_all = dt.to(torch.float32).permute(1, 0, 2).contiguous()
        x_all = x.to(torch.float32)
        C_all = C.index_select(1, head_to_group).to(torch.float32)
        states_all = states.to(torch.float32)
        x_chunks = x_all.new_zeros((nchunks, chunk_size, nheads, headdim))
        C_chunks = C_all.new_zeros((nchunks, chunk_size, nheads, dstate))
        z_chunks = z.to(torch.float32).new_zeros(
            (nchunks, chunk_size, nheads, headdim)) if z is not None else None
        D_fp32 = D.to(torch.float32) if D is not None else None
        cb_all = cb.index_select(1, head_to_group).to(torch.float32)
        row_idx = torch.arange(chunk_size, device=x.device)
        causal = torch.tril(
            torch.ones(chunk_size, chunk_size, device=x.device,
                       dtype=torch.bool))
        valid_rows = row_idx.unsqueeze(0) < chunk_lengths.unsqueeze(1)
        valid_causal = valid_rows[:, :, None] & valid_rows[:, None, :] & causal.unsqueeze(0)

        for chunk_idx in range(nchunks):
            token_start = int(chunk_offsets[chunk_idx].item())
            token_end = int(chunk_offsets[chunk_idx + 1].item())
            if token_end <= token_start:
                continue
            token_count = token_end - token_start
            x_chunks[chunk_idx, :token_count].copy_(x_all[token_start:token_end])
            C_chunks[chunk_idx, :token_count].copy_(C_all[token_start:token_end])
            if z_chunks is not None:
                z_chunks[chunk_idx, :token_count].copy_(z[token_start:token_end].to(torch.float32))

        delta = dA_all.unsqueeze(-1) - dA_all.unsqueeze(-2)
        # For masked future positions, exp(delta) can overflow and then turn
        # into NaN when multiplied by the zero causal mask. Zero those entries
        # before exponentiation and clamp valid deltas to the causal range.
        delta.masked_fill_(~valid_causal.unsqueeze(1), 0)
        delta.clamp_max_(0)
        coeff = cb_all * torch.exp(delta)
        coeff.mul_(dt_all.unsqueeze(-2))
        coeff.mul_(valid_causal.unsqueeze(1))

        x_by_head = x_chunks.permute(0, 2, 1, 3).reshape(-1, chunk_size, headdim)
        acc = torch.bmm(coeff.reshape(-1, chunk_size, chunk_size), x_by_head)
        acc = acc.reshape(nchunks, nheads, chunk_size, headdim).permute(0, 2, 1, 3)

        prev_states = states_all.new_zeros((nchunks, nheads, headdim, dstate))
        if initial_states is not None:
            initial_states_all = initial_states.to(torch.float32)
            prev_states[0] = initial_states_all[seq_ids[0]]
        if nchunks > 1:
            prev_chunk_mask = seq_ids[1:] == seq_ids[:-1]
            prev_chunk_indices = torch.nonzero(prev_chunk_mask, as_tuple=False).flatten()
            if prev_chunk_indices.numel() > 0:
                prev_states[prev_chunk_indices + 1] = states_all[prev_chunk_indices]
            if initial_states is not None:
                boundary_indices = torch.nonzero(~prev_chunk_mask,
                                                as_tuple=False).flatten()
                if boundary_indices.numel() > 0:
                    prev_states[boundary_indices + 1] = initial_states_all[
                        seq_ids[boundary_indices + 1]
                    ]

        prev_term = torch.bmm(
            C_chunks.permute(0, 2, 1, 3).reshape(-1, chunk_size, dstate),
            prev_states.reshape(-1, headdim, dstate).transpose(1, 2).contiguous(),
        )
        prev_term = prev_term.reshape(nchunks, nheads, chunk_size,
                                      headdim).permute(0, 2, 1, 3)
        prev_term.mul_(torch.exp(dA_all).permute(0, 2, 1).unsqueeze(-1))
        acc.add_(prev_term)

        if D_fp32 is not None:
            if D_fp32.dim() == 1:
                acc.add_(x_chunks * D_fp32.view(1, 1, nheads, 1))
            else:
                acc.add_(x_chunks * D_fp32.view(1, 1, nheads, headdim))

        if z_chunks is not None:
            z_gate = z_chunks * torch.sigmoid(z_chunks)
            acc.mul_(z_gate)

        for chunk_idx in range(nchunks):
            token_start = int(chunk_offsets[chunk_idx].item())
            token_end = int(chunk_offsets[chunk_idx + 1].item())
            if token_end <= token_start:
                continue
            token_count = token_end - token_start
            out[token_start:token_end].copy_(acc[chunk_idx, :token_count].to(out.dtype))

        _debug_mamba_io(
            "chunk_scan",
            {
                "device": x.device.type,
                "seqlen": int(seqlen),
                "nheads": int(nheads),
                "headdim": int(headdim),
                "nchunks": int(nchunks),
                "chunk_size": int(chunk_size),
                "has_initial_states": initial_states is not None,
                "result_absmax": float(out.detach().abs().max().cpu()),
                "result_nonfinite": int((~torch.isfinite(out)).sum().detach().cpu()),
            },
        )
        return

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (
        0, 0, 0)
    initial_states_strides = (0, 0, 0, 0)
    initial_states_ptr = states.new_empty((1, 1, 1, 1))

    vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd_kernel[
        grid](
            cb_ptr=cb,
            x_ptr=x,
            z_ptr=z,
            out_ptr=out,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            seq_idx_ptr=seq_idx,
            C_ptr=C,
            states_ptr=states,
            D_ptr=D,
            initstates_ptr=initial_states_ptr,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            chunk_size=chunk_size,
            hdim=headdim,
            dstate=dstate,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_cb_chunk=cb.stride(0),
            stride_cb_head=cb.stride(1),
            stride_cb_csize_m=cb.stride(2),
            stride_cb_csize_k=cb.stride(3),
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_z_seqlen=z_strides[0],
            stride_z_head=z_strides[1],
            stride_z_hdim=z_strides[2],
            stride_out_seqlen=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_hdim=out.stride(2),
            stride_dt_chunk=dt.stride(1),
            stride_dt_head=dt.stride(0),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_seq_idx_chunk=seq_idx.stride(0),
            stride_C_seqlen=C.stride(0),
            stride_C_head=C.stride(1),
            stride_C_dstate=C.stride(2),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=initial_states_strides[0],
            stride_init_states_head=initial_states_strides[1],
            stride_init_states_hdim=initial_states_strides[2],
            stride_init_states_dstate=initial_states_strides[3],
            stride_D_head=D.stride(0) if D is not None else 0,
            IS_CAUSAL=True,
            HAS_D=D is not None,
            D_HAS_HDIM=D.dim() == 2 if D is not None else True,
            HAS_Z=z is not None,
            BLOCK_SIZE_DSTATE=max(
                vllm.model_executor.layers.mamba.ops.ssd_chunk_scan.triton.
                next_power_of_2(dstate), 16),
            IS_TRITON_22=vllm.model_executor.layers.mamba.ops.ssd_chunk_scan.
            TRITON_22,
            HAS_INITSTATES=False,
        )

    return


@torch._dynamo.disable
def _state_passing_fwd_npu(
    states,
    dA_cumsum,
    cu_chunk_seqlens,
    seq_idx,
    initial_states=None,
    out_dtype=None,
):
    if (states.device.type != "npu" or seq_idx is None or states.shape[0] <= 1
            or os.getenv("VLLM_ASCEND_DISABLE_STATE_PASSING_FALLBACK") == "1"):
        return _ORIGINAL_STATE_PASSING_FWD(
            states,
            dA_cumsum,
            cu_chunk_seqlens,
            seq_idx,
            initial_states=initial_states,
            out_dtype=out_dtype,
        )

    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    _debug_mamba_io(
        "state_passing",
        {
            "device": states.device.type,
            "seqlen": int(cu_chunk_seqlens[-1].item()) if cu_chunk_seqlens is not None else None,
            "nchunks": int(nchunks),
            "nheads": int(nheads),
            "dim": int(dim),
            "chunk_size": int(chunk_size),
            "has_initial_states": initial_states is not None,
            "seq_ids_head": seq_idx[: min(8, seq_idx.numel())].detach().cpu().tolist() if seq_idx is not None else None,
            "states_absmax": float(states.detach().abs().max().cpu()),
            "initial_absmax": float(initial_states.detach().abs().max().cpu()) if initial_states is not None else None,
            "dA_last_min": float(dA_cumsum[:, :, -1].detach().min().cpu()),
            "dA_last_max": float(dA_cumsum[:, :, -1].detach().max().cpu()),
        },
    )
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    carry = (
        initial_states[0].to(torch.float32)
        if initial_states is not None
        else torch.zeros((nheads, dim), device=states.device, dtype=torch.float32)
    )
    zero_carry = torch.zeros_like(carry)
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=torch.float32)
    seq_ids = seq_idx.detach().cpu().tolist()

    prev_seq_idx = 0
    for chunk_idx, cur_seq_idx in enumerate(seq_ids):
        if cur_seq_idx != prev_seq_idx:
            carry = (
                initial_states[cur_seq_idx].to(torch.float32)
                if initial_states is not None
                else zero_carry
            )
        prev_seq_idx = cur_seq_idx
        decay = torch.exp(dA_cumsum[:, chunk_idx, -1]).to(torch.float32).unsqueeze(-1)
        carry = decay * carry + states[chunk_idx].to(torch.float32)
        out[chunk_idx] = carry

    _debug_mamba_io(
        "state_passing",
        {
            "device": states.device.type,
            "seqlen": int(cu_chunk_seqlens[-1].item()) if cu_chunk_seqlens is not None else None,
            "nchunks": int(nchunks),
            "nheads": int(nheads),
            "dim": int(dim),
            "chunk_size": int(chunk_size),
            "has_initial_states": initial_states is not None,
            "result_absmax": float(out.detach().abs().max().cpu()),
            "result_nonfinite": int((~torch.isfinite(out)).sum().detach().cpu()),
        },
    )

    return out.to(out_dtype)


for _module in (
        vllm.model_executor.layers.mamba.ops.layernorm_gated,
        vllm.model_executor.layers.mamba.ops.mamba_ssm,
        vllm.model_executor.layers.mamba.ops.ssd_bmm,
        vllm.model_executor.layers.mamba.ops.ssd_chunk_state,
        vllm.model_executor.layers.mamba.ops.ssd_state_passing,
):
    _patch_module_torch(_module)

vllm.model_executor.layers.fused_moe.layer.current_platform = _PlatformProxy(
    vllm.model_executor.layers.fused_moe.layer.current_platform)
_ORIGINAL_CHUNK_SCAN_FWD = (
    vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd)
_ORIGINAL_STATE_PASSING_FWD = (
    vllm.model_executor.layers.mamba.ops.ssd_state_passing._state_passing_fwd)
vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd = (
    _chunk_scan_fwd_npu)
vllm.model_executor.layers.mamba.ops.ssd_state_passing._state_passing_fwd = (
    _state_passing_fwd_npu)
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule
