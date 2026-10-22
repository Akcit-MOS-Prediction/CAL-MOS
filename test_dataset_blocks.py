import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import gerenciador as manager
import grid
from models.base_models import MLPBase, CalMOSDynamicModel, CalMOSDynamicKANModel
from models.dataset_blocks import DatasetBlockExpansion
from models.torch_relu_kan import ReLUKAN
from utils.dataloader import DynamicCollate, DynamicDataset
from utils.dataset_identity import DATASET_IDS, dataset_id_for_split

SPEC = {"enabled": True, "num_blocks": 4, "output_multiplier": 2.5,
        "dataset_to_id": DATASET_IDS}


def check_windows(test, expanded, original, ids, router):
    for row, identifier in enumerate(ids.tolist()):
        start, end = router.windows[identifier - 1]
        torch.testing.assert_close(expanded[row, start:end], original[row], rtol=0, atol=0)
        outside = torch.ones(expanded.shape[1], dtype=torch.bool, device=expanded.device)
        outside[start:end] = False
        test.assertEqual(torch.count_nonzero(expanded[row, outside]).item(), 0)


class FakeBackbone(torch.nn.Module):
    def __init__(self, dimension=8):
        super().__init__()
        self.projection = torch.nn.Linear(1, dimension)

    def forward(self, input_values, **kwargs):
        hidden = self.projection(input_values.unsqueeze(-1))
        return SimpleNamespace(hidden_states=(hidden * 0.5, hidden))


def dynamic_model(head):
    cls = CalMOSDynamicModel if head == "mlp" else CalMOSDynamicKANModel
    with patch("models.base_models.AutoConfig.from_pretrained", return_value=SimpleNamespace()), \
         patch("models.base_models.AutoModel.from_pretrained", return_value=FakeBackbone()):
        return cls(model_name="test-backbone", mlp_input_dim=8, mlp_hidden_dim=16,
                   mlp_output_size=1, mlp_num_layers=1, num_feature_layers=2,
                   last_layer=True, pooling_strategy="mean", mlp_dropout=0,
                   relukan_k=2, relukan_grid=5, use_layer_norm=True,
                   layer_norm_affine=False, freeze_backbone=True,
                   dataset_block_expansion=SPEC)


class DatasetBlockTests(unittest.TestCase):
    def test_all_ids_mixed_batch_shape_zeros_dtype_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            inputs = torch.randn(4, 7, dtype=dtype, requires_grad=True)
            ids = torch.tensor([4, 1, 3, 2])
            router = DatasetBlockExpansion(7, SPEC)
            expanded = router(inputs, ids)
            self.assertEqual(expanded.shape, (4, 18))
            self.assertEqual(expanded.dtype, inputs.dtype)
            check_windows(self, expanded, inputs, ids, router)
            expanded.sum().backward()
            torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
            self.assertEqual(list(router.parameters()), [])

    def test_bad_ids_missing_ids_shapes_and_changed_mapping_are_rejected(self):
        router = DatasetBlockExpansion(7)
        for ids in (None, torch.tensor([0, 1]), torch.tensor([1, 5]),
                    torch.tensor([1., 2.]), torch.tensor([[1], [2]]), torch.tensor([1])):
            with self.assertRaises(ValueError):
                router(torch.randn(2, 7), ids)
        with self.assertRaises(ValueError):
            router(torch.randn(2, 8), torch.tensor([1, 2]))
        with self.assertRaises(ValueError):
            DatasetBlockExpansion(7, {"dataset_to_id": {**DATASET_IDS, "bvcc": 4}})

    def test_mhubert_768_embeddings_are_zero_padded_into_four_1024_blocks(self):
        specification = {**SPEC, "output_multiplier": 4.0, "block_width": 1024}
        router = DatasetBlockExpansion(768, specification)
        inputs = torch.randn(4, 768)
        ids = torch.tensor([1, 2, 3, 4])
        expanded = router(inputs, ids)
        self.assertEqual(expanded.shape, (4, 4096))
        self.assertEqual(router.start_positions, (0, 1024, 2048, 3072))
        for row, identifier in enumerate(ids.tolist()):
            start = (identifier - 1) * 1024
            torch.testing.assert_close(expanded[row, start:start + 768], inputs[row], rtol=0, atol=0)
            torch.testing.assert_close(expanded[row, start + 768:start + 1024],
                                       torch.zeros(256), rtol=0, atol=0)
            outside = torch.ones(4096, dtype=torch.bool)
            outside[start:start + 1024] = False
            self.assertEqual(torch.count_nonzero(expanded[row, outside]).item(), 0)
        clone = DatasetBlockExpansion(768, specification)
        clone.load_state_dict(router.state_dict(), strict=True)
        self.assertEqual(router.get_extra_state()["version"], 4)
        wrong = copy.deepcopy(router.state_dict())
        wrong["_extra_state"]["block_width"] = 768
        with self.assertRaises(ValueError):
            clone.load_state_dict(wrong, strict=True)
    def test_mapping_and_dimension_are_validated_on_checkpoint_reload(self):
        router = DatasetBlockExpansion(7)
        state = router.state_dict()
        clone = DatasetBlockExpansion(7)
        clone.load_state_dict(state, strict=True)
        for field, changed in (("input_size", 8),
                               ("dataset_to_id", {**DATASET_IDS, "bvcc": 4}),
                               ("output_multiplier", 4.0)):
            wrong = copy.deepcopy(state)
            wrong["_extra_state"][field] = changed
            with self.assertRaises(ValueError):
                clone.load_state_dict(wrong, strict=True)

    def test_compact_and_legacy_four_x_layouts_are_integer_and_checkpoint_stable(self):
        compact = DatasetBlockExpansion(1024, {**SPEC, "output_multiplier": 2.5})
        self.assertEqual(compact.output_size, 2560)
        self.assertEqual(compact.start_positions, (0, 512, 1024, 1536))
        self.assertEqual(compact.windows,
                         ((0, 1024), (512, 1536), (1024, 2048), (1536, 2560)))

        near_26 = DatasetBlockExpansion(1024, {**SPEC, "output_multiplier": 2.6})
        self.assertEqual(near_26.output_size, 2662)
        self.assertEqual(near_26.start_positions, (0, 546, 1092, 1638))
        self.assertEqual(near_26.windows[-1], (1638, 2662))

        legacy = DatasetBlockExpansion(1024)
        self.assertEqual(legacy.output_size, 4096)
        self.assertEqual(legacy.start_positions, (0, 1024, 2048, 3072))
        self.assertEqual(legacy.get_extra_state(), {
            "version": 1, "input_size": 1024, "num_blocks": 4,
            "dataset_to_id": dict(DATASET_IDS),
        })
        DatasetBlockExpansion(1024, {**SPEC, "output_multiplier": 4.0}).load_state_dict(
            legacy.state_dict(), strict=True
        )

    def test_one_x_is_original_embedding_for_every_dataset(self):
        router = DatasetBlockExpansion(8, {**SPEC, "output_multiplier": 1.0})
        inputs = torch.randn(4, 8)
        ids = torch.tensor([1, 2, 3, 4])
        expanded = router(inputs, ids)
        self.assertEqual(expanded.shape, inputs.shape)
        self.assertEqual(router.start_positions, (0, 0, 0, 0))
        torch.testing.assert_close(expanded, inputs, rtol=0, atol=0)

    def test_invalid_multipliers_and_too_small_output_are_rejected(self):
        for multiplier in (True, float("nan"), float("inf"), 0.99, 1.1):
            with self.assertRaises(ValueError):
                DatasetBlockExpansion(8, {**SPEC, "output_multiplier": multiplier})

    def test_nonzero_padding_value_fills_only_positions_outside_dataset_window(self):
        specification = {**SPEC, "output_multiplier": 2.0, "padding_value": 0.1}
        router = DatasetBlockExpansion(8, specification)
        inputs = torch.randn(4, 8)
        ids = torch.tensor([1, 2, 3, 4])
        expanded = router(inputs, ids)
        self.assertEqual(router.output_size, 16)
        self.assertEqual(router.start_positions, (0, 3, 5, 8))
        for row, identifier in enumerate(ids.tolist()):
            start, end = router.windows[identifier - 1]
            outside = torch.ones(16, dtype=torch.bool)
            outside[start:end] = False
            torch.testing.assert_close(expanded[row, outside], torch.full((8,), 0.1))
            torch.testing.assert_close(expanded[row, start:end], inputs[row])

    def test_expansion_is_after_original_non_affine_norm_in_both_heads(self):
        heads = [MLPBase(input_size=8, hidden_dim=16, output_size=1, dropout=0,
                         dataset_block_expansion=SPEC),
                 ReLUKAN([8, 16, 1], 5, 2, dataset_block_expansion=SPEC)]
        inputs, ids = torch.randn(4, 8), torch.tensor([1, 2, 3, 4])
        for head in heads:
            first = head.layers[0] if isinstance(head, MLPBase) else head.rk_layers[0]
            captured = []
            hook = first.register_forward_pre_hook(lambda module, values: captured.append(values[0]))
            output = head(inputs, ids)
            hook.remove()
            self.assertEqual(output.shape, (4, 1))
            self.assertFalse(head.input_layer_norm.elementwise_affine)
            self.assertEqual(head.input_layer_norm.normalized_shape, (8,))
            check_windows(self, captured[0], head.input_layer_norm(inputs), ids,
                          head.block_expansion)
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))

    def test_affine_or_missing_norm_is_rejected(self):
        for use_norm, affine in ((False, False), (True, True)):
            for constructor in (
                lambda: MLPBase(input_size=8, dataset_block_expansion=SPEC,
                                use_layer_norm=use_norm, layer_norm_affine=affine),
                lambda: ReLUKAN([8, 16, 1], 5, 2, dataset_block_expansion=SPEC,
                               use_layer_norm=use_norm, layer_norm_affine=affine),
            ):
                with self.assertRaises(ValueError):
                    constructor()

    def test_loader_attaches_dataset_id_and_preserves_legacy_examples(self):
        data = pd.DataFrame({"file": ["example.wav"], "mos": [2.5]})
        for identifier in (None, 1, 2, 3, 4):
            dataset = DynamicDataset(data, Path("/unused"), "file", "mos", dataset_id=identifier)
            with patch.object(dataset, "_load_wav", return_value=(torch.ones(1, 32), 16000)):
                example = dataset[0]
            self.assertEqual(len(example), 3 if identifier is None else 4)
            if identifier is not None:
                self.assertEqual(example[-1], identifier)
        with self.assertRaises(ValueError):
            DynamicDataset(data, Path("/unused"), "file", "mos", dataset_id=0)

    def test_collate_keeps_dataset_and_sampling_rate_ids_separate(self):
        processor = lambda audios, **kw: {"input_values": torch.tensor(np.array(audios), dtype=torch.float32)}
        collate = DynamicCollate(processor=processor)
        examples = [(np.ones(32), 2, 2.5, 1), (np.zeros(32), 0, 3.5, 4)]
        features, targets = collate(examples)
        torch.testing.assert_close(features[1], torch.tensor([2, 0]))
        torch.testing.assert_close(features[2], torch.tensor([1, 4]))
        torch.testing.assert_close(targets, torch.tensor([2.5, 3.5]))
        legacy, _ = collate([row[:3] for row in examples])
        self.assertEqual(len(legacy), 2)
        with self.assertRaises(ValueError):
            collate([examples[0], examples[1][:3]])

    def test_split_identity_uses_actual_dataset_not_checkpoint_origin(self):
        config = OmegaConf.create({"model": {"dataset_block_expansion": SPEC}})
        self.assertEqual(dataset_id_for_split(config, {"dataset_name": "bvcc"}), 1)
        self.assertEqual(dataset_id_for_split(config, {"dataset_name": "brspeech"}), 4)
        with self.assertRaises(ValueError):
            dataset_id_for_split(config, {})
        config.model.dataset_block_expansion.enabled = False
        self.assertIsNone(dataset_id_for_split(config, {}))

    def test_older_dispatcher_metadata_tags_route_correctly_and_fail_if_ambiguous(self):
        config = OmegaConf.create({"model": {"dataset_block_expansion": SPEC},
                                  "tags": ["grid", "cross-dataset", "kan", "brspeech"]})
        self.assertEqual(dataset_id_for_split(config, {}), 4)
        self.assertEqual(dataset_id_for_split(config, {"dataset_name": "bvcc"}), 1)
        config.tags.append("bvcc")
        with self.assertRaises(ValueError):
            dataset_id_for_split(config, {})

    def test_inference_preserves_mixed_dataset_ids(self):
        from eval.inference import inference
        features = ({"input_values": torch.randn(4, 16)},
                    torch.zeros(4, dtype=torch.long), torch.tensor([1, 2, 3, 4]))
        targets = torch.tensor([1., 2., 3., 4.])
        for head in ("mlp", "kan"):
            model = dynamic_model(head).eval()
            expected = model(features).detach().numpy()
            predicted, observed = inference(model, [(features, targets)], torch.device("cpu"))
            np.testing.assert_allclose(predicted, expected, rtol=0, atol=0)
            np.testing.assert_array_equal(observed, targets.numpy())

    def test_dynamic_source_training_target_finetuning_and_reload(self):
        torch.manual_seed(123)
        features = {"input_values": torch.randn(4, 16)}
        for head in ("mlp", "kan"):
            model = dynamic_model(head)
            self.assertEqual(model._get_embedding_dim(), 8)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            for identifier in (1, 1, 4, 4):
                prediction = model((features, torch.zeros(4, dtype=torch.long), torch.full((4,), identifier)))
                loss = (prediction - torch.tensor([1., 2., 3., 4.])).square().mean()
                self.assertTrue(torch.isfinite(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = dynamic_model(head)
            restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
            restored.eval()
            first = restored.mlp.layers[0] if head == "mlp" else restored.mlp.rk_layers[0]
            captured = []
            hook = first.register_forward_pre_hook(lambda module, values: captured.append(values[0]))
            for identifier in (1, 4):
                ids = torch.full((4,), identifier)
                output = restored((features, torch.zeros(4, dtype=torch.long), ids))
                self.assertEqual(output.shape, (4,))
                self.assertTrue(torch.isfinite(output).all())
                normalized = restored.mlp.input_layer_norm(restored.backbone(**features).hidden_states[-1].mean(1))
                check_windows(self, captured[-1], normalized, ids,
                              restored.mlp.block_expansion)
            hook.remove()
            with self.assertRaises(ValueError):
                restored((features, torch.zeros(4, dtype=torch.long)))

    def test_unexpanded_heads_load_previous_checkpoint_and_match_predictions(self):
        original_root = Path(os.environ.get("CALMOS_ORIGINAL_ROOT", manager.ROOT))
        for filename, cls, args in (
            ("base_models.py", "MLPBase", {"input_size": 8, "hidden_dim": 16, "output_size": 1, "dropout": 0}),
            ("torch_relu_kan.py", "ReLUKAN", {"width": [8, 16, 1], "grid": 5, "k": 2}),
        ):
            spec = importlib.util.spec_from_file_location("previous_" + cls, original_root / "src/models" / filename)
            previous = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(previous)
            old = getattr(previous, cls)(**args).eval()
            new = (MLPBase if cls == "MLPBase" else ReLUKAN)(**args).eval()
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            new.load_state_dict(old.state_dict(), strict=True)
            inputs = torch.randn(4, 8)
            torch.testing.assert_close(old(inputs), new(inputs), rtol=0, atol=0)

    def test_grid_all_sources_and_finetunes_keep_mapping_and_split_ids(self):
        config_dir = Path(__file__).resolve().parent / "config/models"
        real_path = manager.config_path
        def config_path(name, kind):
            if kind == "models" and name.startswith("wavlm_large_dataset_blocks"):
                return config_dir / (name + ".yaml")
            return real_path(name, kind)
        with tempfile.TemporaryDirectory(prefix="calmos-block-grid-test-") as temporary:
            root = Path(temporary)
            with patch.object(manager, "config_path", side_effect=config_path), \
                 patch.object(manager, "QUEUE_DIR", root / "fila"), \
                 patch.object(manager, "RESULTS_DIR", root / "resultados"), \
                 patch.object(manager, "DB_PATH", root / "fila/runs.sqlite3"), \
                 patch.object(grid, "EXPERIMENTS_DIR", root / "resultados/grid"), \
                 patch.object(manager, "git_revision", return_value={}):
                connection = manager.database()
                try:
                    identifier = grid.create_experiment(connection, 16, 3, 10,
                        layer_norm_affine=False, gpu_pool=(5, 6, 7), config_model="wavlm_large_dataset_blocks")
                    plan = json.loads((root / "resultados/grid" / identifier / "grid.json").read_text())
                    self.assertEqual(plan["dataset_block_expansion"]["output_multiplier"], 2.5)
                    self.assertEqual(plan["dataset_block_expansion"]["head_input_dim"], 2560)
                    self.assertEqual(plan["dataset_block_expansion"]["window_starts"], [0, 512, 1024, 1536])
                    sources = grid.jobs_for(connection, identifier)
                    self.assertEqual(len(sources), 8)
                    for source in sources:
                        folder = Path(source["result_dir"])
                        checkpoint = folder / "checkpoints/best.ckpt"
                        checkpoint.write_bytes(b"queue-only-checkpoint")
                        manager.write_json(folder / "grid_result.json", {
                            "best_checkpoint": str(checkpoint),
                            "tests": {source["source"]: {"srcc": .7, "mse": .3}},
                        })
                        connection.execute("UPDATE runs SET status='succeeded' WHERE id=?", (source["run_id"],))
                    grid.advance_ready(connection)
                    jobs = grid.jobs_for(connection, identifier)
                    self.assertEqual(len(jobs), 32)
                    for job in jobs:
                        config = OmegaConf.load(Path(job["result_dir"]) / "config.yaml")
                        dataset = job["target"] or job["source"]
                        for split in ("train", "val", "test"):
                            self.assertEqual(dataset_id_for_split(config, config.datasets[split][0]), DATASET_IDS[dataset])
                        self.assertEqual(config.model.mlp_input_dim, 1024)
                        self.assertEqual(config.model.mlp_hidden_dim, 1024)
                        self.assertEqual((config.model.relukan_k, config.model.relukan_grid), (2, 5))
                        self.assertFalse(config.model.layer_norm_affine)
                        self.assertEqual(config.model.dataset_block_expansion.output_multiplier, 2.5)
                        self.assertIn("blocos2p5", job["result_dir"])
                        if job["kind"] == "finetune":
                            manifest = json.loads((Path(job["result_dir"]) / "grid_job.json").read_text())
                            origin = OmegaConf.load(manifest["source_config"])
                            self.assertEqual(dataset_id_for_split(origin, origin.datasets.test[0]), DATASET_IDS[job["source"]])
                    for gpu in (0, 1, 2, 3, 4):
                        self.assertIsNone(manager.claim(connection, gpu))
                    for gpu in (5, 6, 7):
                        self.assertIsNotNone(manager.claim(connection, gpu))
                finally:
                    connection.close()

    def test_sequential_chain_queues_base_training_then_ordered_finetunings(self):
        with tempfile.TemporaryDirectory(prefix="calmos-sequential-chain-test-") as temporary:
            root = Path(temporary)
            with (patch.object(manager, "QUEUE_DIR", root / "fila"),
                  patch.object(manager, "RESULTS_DIR", root / "resultados"),
                  patch.object(manager, "DB_PATH", root / "fila" / "runs.sqlite3"),
                  patch.object(manager, "git_revision", return_value={})):
                connection = manager.database()
                try:
                    rows = grid.create_sequential_chains(
                        connection, name="test-blocks-4x",
                        config_model="mHuBERT-147", heads=("kan", "mlp"),
                        sequence=("bvcc", "brspeech", "singmos", "tmhint"),
                        gpu_pool=(0, 2), relukan_grid=5,
                    )
                    self.assertEqual([(row["head"], row["gpu"]) for row in rows],
                                     [("kan", 0), ("mlp", 2)])
                    for row in rows:
                        run_dir = Path(row["result_dir"])
                        plan = json.loads((run_dir / "forgetting.json").read_text())
                        self.assertEqual(plan["sequence"], ["bvcc", "brspeech", "singmos", "tmhint"])
                        self.assertIsNone(plan["initial_checkpoint"])
                        self.assertIsNone(plan["evaluation_datasets"])
                        self.assertEqual(set(plan["evaluation_configs"]),
                                         {"bvcc", "brspeech", "singmos", "tmhint"})
                        self.assertIn("blocos4", row["result_dir"])
                        for stage in plan["stages"]:
                            config = manager.load_yaml(run_dir / stage["config"])
                            self.assertEqual(config["model"]["dataset_block_expansion"]["output_multiplier"], 4.0)
                            self.assertFalse(config["model"]["layer_norm_affine"])
                            self.assertEqual(config["model"]["mlp_input_dim"], 768)
                            self.assertEqual(config["model"]["relukan_k"], 2)
                            self.assertEqual(config["model"]["relukan_grid"], 5)
                finally:
                    connection.close()


if __name__ == "__main__":
    unittest.main()
