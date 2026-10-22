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

    def test_first_layer_receives_sigmoid_of_layernorm_with_padding_below_grid(self):
        head = ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)
        inputs, ids = torch.randn(4, 8) * 50, torch.tensor([1, 2, 3, 4])
        captured, handles = capture_inputs(head)
        head(inputs, ids)
        for handle in handles:
            handle.remove()
        first = captured[0]
        expected = torch.sigmoid(F.layer_norm(inputs, (8,)))
        self.assertEqual(head.block_expansion.padding_value, -1.0)
        for row, identifier in enumerate(ids.tolist()):
            start, end = head.block_expansion.windows[identifier - 1]
            torch.testing.assert_close(first[row, start:end], expected[row], rtol=0, atol=0)
            window = first[row, start:end]
            self.assertTrue(((window > 0) & (window < 1)).all())
            outside = torch.ones(first.shape[1], dtype=torch.bool)
            outside[start:end] = False
            self.assertTrue((first[row, outside] == -1.0).all())
            # Every real coordinate activates at least one basis; every padded one activates none.
            activity = grid_activity(head.rk_layers[0], first[row:row + 1])[0]
            self.assertTrue((activity[start:end] > 0).all())
            self.assertTrue((activity[outside] == 0).all())

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

    def test_padded_positions_get_exactly_zero_gradient_and_no_influence(self):
        head = ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion=BLOCKS)
        inputs, ids = torch.randn(5, 8), torch.full((5,), 2)
        head(inputs, ids).square().sum().backward()
        gradient = head.rk_layers[0].equal_size_conv.weight.grad  # [out, 1, g+k, 4*8]
        start, end = head.block_expansion.windows[1]
        padded = torch.ones(gradient.shape[-1], dtype=torch.bool)
        padded[start:end] = False
        self.assertEqual(torch.count_nonzero(gradient[..., padded]).item(), 0)
        self.assertGreater(torch.count_nonzero(gradient[..., ~padded]).item(), 0)
        before = head(inputs, ids).detach()
        with torch.no_grad():
            head.rk_layers[0].equal_size_conv.weight[..., padded] = torch.randn_like(
                head.rk_layers[0].equal_size_conv.weight[..., padded])
        torch.testing.assert_close(head(inputs, ids), before, rtol=0, atol=0)

    def test_padding_inside_grid_is_rejected_and_default_follows_the_grid(self):
        for padding in (0.0, 0.5, 1.0, -0.4, 1.4):
            with self.assertRaisesRegex(ValueError, "inside the ReLU-KAN grid"):
                ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion={**BLOCKS, "padding_value": padding})
        self.assertEqual(ReLUKAN([8, 16, 1], grid=5, k=2, dataset_block_expansion={
            **BLOCKS, "padding_value": 2.0}).block_expansion.padding_value, 2.0)
        # g=3, k=3 -> grid [-1, 2]; -1.0 would sit on the grid edge, so the default moves below it.
        self.assertEqual(ReLUKAN([8, 16, 1], grid=3, k=3, dataset_block_expansion=BLOCKS)
                         .block_expansion.padding_value, -1.5)

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
        with self.assertRaisesRegex(RuntimeError, "predates the sigmoid"):
            clone.load_state_dict(legacy, strict=True)
        wrong = dict(state)
        wrong["_extra_state"] = {"version": 1, "input": "layernorm"}
        with self.assertRaises(ValueError):
            clone.load_state_dict(wrong, strict=True)


if __name__ == "__main__":
    unittest.main()
