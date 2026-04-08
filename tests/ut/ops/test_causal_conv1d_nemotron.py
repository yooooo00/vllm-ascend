from unittest import TestCase
from unittest.mock import patch

import torch

from vllm_ascend.ops.triton.mamba import causal_conv1d as causal_conv1d_mod


class _FakePCPGroup:
    world_size = 1


class _FakeForwardContext:
    attn_metadata = None


class TestNemotronCausalConv1d(TestCase):

    def test_accepts_apc_kwargs_and_respects_has_initial_state(self):
        captured = []

        def _fake_extract_last_width(x, start_loc, width):
            return torch.zeros(
                (start_loc.numel() - 1, x.shape[0], width),
                dtype=x.dtype,
                device=x.device,
            )

        def _fake_ref(
            x,
            weight,
            bias=None,
            initial_states=None,
            return_final_states=False,
            final_states_out=None,
            activation=None,
        ):
            captured.append(initial_states.clone() if initial_states is not None
                            else None)
            if return_final_states and final_states_out is not None:
                final_states_out.copy_(torch.full_like(final_states_out, 7))
            return x.unsqueeze(0), final_states_out

        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        weight = torch.ones((4, 3), dtype=torch.float32)
        conv_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        initial_slice = conv_states[0, :, :2].clone()
        has_initial_state = torch.tensor([True, False], dtype=torch.bool)
        cache_indices = torch.tensor([0, 1], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)

        with patch.object(causal_conv1d_mod, "get_pcp_group",
                          return_value=_FakePCPGroup()), \
                patch.object(causal_conv1d_mod,
                             "get_forward_context",
                             return_value=_FakeForwardContext()), \
                patch.object(causal_conv1d_mod,
                             "extract_last_width",
                             side_effect=_fake_extract_last_width), \
                patch.object(causal_conv1d_mod,
                             "causal_conv1d_ref",
                             side_effect=_fake_ref):
            out = causal_conv1d_mod.causal_conv1d_fn(
                x,
                weight,
                conv_states=conv_states,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                block_idx_first_scheduled_token=torch.tensor(
                    [0, 0], dtype=torch.int32),
                block_idx_last_scheduled_token=torch.tensor(
                    [1, 1], dtype=torch.int32),
                initial_state_idx=torch.tensor([0, 1], dtype=torch.int32),
                num_computed_tokens=torch.tensor([0, 0], dtype=torch.int32),
                block_size_to_align=16,
                metadata={"unused": True},
            )

        self.assertEqual(out.shape, (1, 4, 4))
        torch.testing.assert_close(captured[0], initial_slice)
        self.assertIsNone(captured[1])
        torch.testing.assert_close(conv_states[0, :, :2],
                                   torch.full_like(conv_states[0, :, :2], 7))
        torch.testing.assert_close(conv_states[1, :, :2],
                                   torch.full_like(conv_states[1, :, :2], 7))
