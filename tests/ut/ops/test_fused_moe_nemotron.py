from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE, AscendSharedFusedMoE
from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp


class _UpProjOnlySharedExperts:

    def __init__(self):
        self.calls = 0

    def up_proj(self, hidden_states):
        self.calls += 1
        return hidden_states + 1, None


class TestNemotronSharedExperts(TestCase):

    def test_init_seeds_gate_before_parent_runner_init(self):
        shared_experts = object()
        gate = object()
        observed = {}

        def fake_init_runner(self):
            observed["gate"] = self.gate
            observed["shared_experts"] = self._shared_experts
            return "runner"

        def fake_parent_init(self, **kwargs):
            self.moe_config = SimpleNamespace(hidden_dim=kwargs["hidden_size"])
            self.quant_method = SimpleNamespace(process_weights_after_loading=lambda *a, **k: None)
            self.runner = self._init_runner()

        fake_cfg = SimpleNamespace(
            mix_placement=False,
            multistream_overlap_shared_expert=False,
            multistream_overlap_gate=False,
        )

        with patch("vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config", return_value=fake_cfg), \
                patch("vllm_ascend.ops.fused_moe.fused_moe.enable_sp", return_value=False), \
                patch.object(AscendFusedMoE, "__init__", fake_parent_init), \
                patch.object(AscendSharedFusedMoE, "_init_runner", fake_init_runner):
            layer = AscendSharedFusedMoE(
                shared_experts=shared_experts,
                gate=gate,
                hidden_size=16,
                intermediate_size=32,
                params_dtype=torch.bfloat16,
                reduce_results=False,
                renormalize=False,
                use_grouped_topk=False,
                num_experts=8,
                top_k=2,
                quant_config=None,
                prefix="test",
            )

        self.assertIs(observed["gate"], gate)
        self.assertIs(observed["shared_experts"], shared_experts)
        self.assertEqual(layer.runner, "runner")

    def test_init_runner_prefers_seeded_shared_experts_before_runner_property(self):
        layer = object.__new__(AscendFusedMoE)
        layer.moe_config = SimpleNamespace()
        layer.router = object()
        layer._routed_input_transform = object()
        layer.quant_method = object()
        layer.reduce_results = False
        layer.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_dbo=False))
        layer._shared_experts = object()
        layer._gate = object()

        observed = {}

        def fake_runner_init(*args, **kwargs):
            observed.update(kwargs)

        with patch("vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner", autospec=True) as runner_cls:
            runner_cls.side_effect = fake_runner_init
            AscendFusedMoE._init_runner(layer)

        self.assertIs(observed["shared_experts"], layer._shared_experts)
        self.assertIs(observed["gate"], layer._gate)

    def test_unquantized_method_reports_modular_before_backend_kernel_init(self):
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

        def fake_parent_init(self, moe=None):
            torch.nn.Module.__init__(self)
            self.moe = moe
            self.experts_cls = None

        fake_cfg = SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False))
        with patch.object(UnquantizedFusedMoEMethod, "__init__", fake_parent_init),                 patch("vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config", return_value=fake_cfg):
            method = AscendUnquantizedFusedMoEMethod(moe=SimpleNamespace())

        self.assertFalse(method.is_monolithic)

    def test_shared_experts_part1_falls_back_to_up_proj(self):
        layer = object.__new__(AscendSharedFusedMoE)
        shared_experts = _UpProjOnlySharedExperts()
        layer._shared_experts = shared_experts

        hidden_states = torch.randn(2, 4)
        out = AscendSharedFusedMoE._shared_experts_part1(layer, hidden_states)

        self.assertEqual(shared_experts.calls, 1)
        torch.testing.assert_close(out, hidden_states + 1)

    def test_forward_impl_uses_shared_experts_input_helper(self):
        layer = object.__new__(AscendSharedFusedMoE)
        layer._shared_experts = object()
        layer.multistream_overlap_gate = False

        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 8)
        shared_input = hidden_states + 3
        recorded = {}

        layer._get_shared_experts_input = lambda hs: shared_input

        def _forward_shared_experts(hs, events):
            recorded["hidden_states"] = hs
            recorded["events"] = events
            return "shared"

        layer._forward_shared_experts = _forward_shared_experts

        with patch.object(
            AscendFusedMoE,
            "forward_impl",
            return_value=SimpleNamespace(
                routed_out="routed",
                before_dispatch_evt=None,
                before_combine_evt=None,
            ),
        ):
            shared_out, routed_out = AscendSharedFusedMoE.forward_impl(
                layer, hidden_states, router_logits)

        self.assertEqual(routed_out, "routed")
        self.assertEqual(shared_out, "shared")
        self.assertIs(recorded["hidden_states"], shared_input)

    def test_unquant_apply_mlp_accepts_relu2_enum_without_swiglu(self):
        grouped_outputs = [
            torch.tensor([[2.0, -3.0]], dtype=torch.float32),
            torch.tensor([[4.0, 9.0]], dtype=torch.float32),
        ]

        def fake_grouped_matmul(*args, **kwargs):
            return [grouped_outputs.pop(0)]

        with patch("vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                   side_effect=fake_grouped_matmul), \
             patch("vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                   side_effect=AssertionError("relu2_no_mul path must not call swiglu")):
            out = unquant_apply_mlp(
                hidden_states=torch.randn(1, 2),
                w1=torch.randn(1, 2, 2),
                w2=torch.randn(1, 2, 2),
                group_list=torch.tensor([1], dtype=torch.int64),
                activation=MoEActivation.RELU2_NO_MUL,
                group_list_type=1,
                need_trans=False,
            )

        torch.testing.assert_close(out, torch.tensor([[4.0, 9.0]], dtype=torch.float32))
