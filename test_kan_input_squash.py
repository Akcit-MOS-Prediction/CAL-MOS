"""ReLU-KAN inputs stay inside the basis grid; the MLP head keeps LayerNorm only.

Spec: specs/kan-input-sigmoid.md. Run from the repo root with PYTHONPATH=src:
    python -m unittest test_kan_input_squash -v
"""
import unittest

import torch
import torch.nn.functional as F

from models.base_models import MLPBase
from models.torch_relu_kan import ReLUKAN
from utils.dataset_identity import DATASET_IDS

BLOCKS = {"enabled": True, "num_blocks": 4, "output_multiplier": 4.0, "dataset_to_id": DATASET_IDS}


def grid_activity(layer, values):
    """Number of active bases per input coordinate of a ReLUKANLayer."""
    inside = (values.unsqueeze(-1) > layer.phase_low) & (values.unsqueeze(-1) < layer.phase_height)
    return inside.sum(-1)


def capture_inputs(head):
    captured = {}
    handles = [layer.register_forward_pre_hook(
        lambda module, args, index=index: captured.__setitem__(index, args[0].detach()))
        for index, layer in enumerate(head.rk_layers)]
    return captured, handles


class KanInputSquashTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_first_layer_receives_sigmoid_of_layernorm_with_zero_padding(self):
        head = ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)
        inputs, ids = torch.randn(4, 8) * 50, torch.tensor([1, 2, 3, 4])
        captured, handles = capture_inputs(head)
        head(inputs, ids)
        for handle in handles:
            handle.remove()
        first = captured[0]
        expected = torch.sigmoid(F.layer_norm(inputs, (8,)))
        self.assertEqual(head.block_expansion.padding_value, 0.0)
        for row, identifier in enumerate(ids.tolist()):
            start, end = head.block_expansion.windows[identifier - 1]
            torch.testing.assert_close(first[row, start:end], expected[row], rtol=0, atol=0)
            window = first[row, start:end]
            self.assertTrue(((window > 0) & (window < 1)).all())
            outside = torch.ones(first.shape[1], dtype=torch.bool)
            outside[start:end] = False
            self.assertTrue((first[row, outside] == 0.0).all())
            # Every real coordinate activates at least one basis.
            activity = grid_activity(head.rk_layers[0], first[row:row + 1])[0]
            self.assertTrue((activity[start:end] > 0).all())

    def test_inner_layers_receive_sigmoid_of_layernorm_even_when_activations_grow(self):
        head = ReLUKAN([8, 16, 16, 1], grid=5, k=2)
        with torch.no_grad():
            for layer in head.rk_layers:
                layer.equal_size_conv.weight.mul_(100.0)
        captured, handles = capture_inputs(head)
        head(torch.randn(6, 8))
        for handle in handles:
            handle.remove()
        for index in (1, 2):
            values = captured[index]
            self.assertTrue(((values > 0) & (values < 1)).all(), f"layer {index} input outside (0, 1)")
            self.assertTrue((grid_activity(head.rk_layers[index], values) > 0).all())

    def test_padding_defaults_to_zero_like_the_mlp_and_honors_configured_values(self):
        self.assertEqual(ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)
                         .block_expansion.padding_value, 0.0)
        self.assertEqual(MLPBase(8, 16, 1, 1, dataset_block_expansion=BLOCKS)
                         .block_expansion.padding_value, 0.0)
        for padding in (0.0, 0.1, -1.0):
            head = ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion={**BLOCKS, "padding_value": padding})
            self.assertEqual(head.block_expansion.padding_value, padding)

    def test_hidden_width_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hidden widths must be >= 2"):
            ReLUKAN([8, 1, 1], grid=5, k=2)

    def test_mlp_head_keeps_layernorm_only_and_zero_padding(self):
        head = MLPBase(input_size=8, hidden_dim=16, num_layers=1, output_size=1, dropout=0,
                       dataset_block_expansion=BLOCKS).eval()
        inputs, ids = torch.randn(4, 8) * 50, torch.tensor([4, 3, 2, 1])
        captured = []
        handle = head.layers[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        output = head(inputs, ids)
        handle.remove()
        normalized = F.layer_norm(inputs, (8,))
        expected = torch.zeros(4, 32)
        for row, identifier in enumerate(ids.tolist()):
            start, end = head.block_expansion.windows[identifier - 1]
            expected[row, start:end] = normalized[row]
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(output, head.layers(expected), rtol=0, atol=0)
        self.assertFalse(hasattr(head, "rk_layers"))

    def test_parameter_counts_are_unchanged(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(count(ReLUKAN([1024, 1024, 1], grid=5, k=2)), 7_376_897)
        self.assertEqual(count(ReLUKAN([1024, 1024, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)), 29_440_001)
        self.assertEqual(count(MLPBase(1024, 1024, 1, 1, dataset_block_expansion=BLOCKS)), 4_196_353)

    def test_checkpoint_roundtrip_and_pre_sigmoid_checkpoint_rejected(self):
        head = ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)
        state = head.state_dict()
        self.assertEqual(state["_extra_state"]["input"], "layernorm+sigmoid")
        clone = ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)
        clone.load_state_dict(state, strict=True)
        legacy = {key: value for key, value in state.items() if key != "_extra_state"}
        for strict in (True, False):
            with self.assertRaisesRegex(RuntimeError, "predates the sigmoid"):
                clone.load_state_dict(legacy, strict=strict)
        # The real case: a feat/4times 4x KAN checkpoint (no head marker, blocks padded with 0).
        legacy_blocks = dict(legacy)
        legacy_blocks["block_expansion._extra_state"] = {
            "version": 1, "input_size": 8, "num_blocks": 4, "dataset_to_id": dict(DATASET_IDS)}
        with self.assertRaisesRegex(RuntimeError, "predates the sigmoid"):
            clone.load_state_dict(legacy_blocks, strict=True)
        wrong = dict(state)
        wrong["_extra_state"] = {"version": 1, "input": "layernorm"}
        with self.assertRaises(ValueError):
            clone.load_state_dict(wrong, strict=True)


if __name__ == "__main__":
    unittest.main()
