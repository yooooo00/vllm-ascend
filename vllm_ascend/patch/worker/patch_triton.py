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
    if x.device.type != "npu" or initial_states is not None:
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
vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd = (
    _chunk_scan_fwd_npu)
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule
