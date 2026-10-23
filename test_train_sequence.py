"""CPU-only integration tests: real heads/Lightning, synthetic tiny backbone/audio."""
import csv
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import train_sequence as sequence
from utils.sequence_outputs import csv_write, result_rows
from utils.dataset_identity import DATASET_IDS


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 8)

    def forward(self, input_values, **kwargs):
        # Temporal variation + nonlinear features prevent all samples having the same LN direction.
        values = input_values.unsqueeze(-1)
        hidden = torch.tanh(self.projection(values))
        return SimpleNamespace(hidden_states=(hidden * 0.5, hidden))


def processor(audios, **kwargs):
    return {"input_values": torch.tensor(np.array(audios), dtype=torch.float32)}


class SequenceTests(unittest.TestCase):
    def fixture(self, root, head="kan"):
        datasets = {}
        for dataset in DATASET_IDS:
            base = root / dataset
            base.mkdir(exist_ok=True)
            (base / "audio.wav").touch()
            splits = {}
            for name in ("train", "val", "test"):
                metadata = base / (name + ".csv")
                metadata.write_text("filepath,mos\naudio.wav,1\naudio.wav,2\naudio.wav,3\naudio.wav,4\n")
                splits[name] = [{"name": name, "metadata_path": str(metadata), "base_dir": str(base),
                                 "filename_column": "filepath", "target_column": "mos"}]
            path = root / f"{dataset}.yaml"
            OmegaConf.save(OmegaConf.create({"datasets": splits, "data": {
                "num_classes": 1, "target_sr": 16000, "mixup_alpha": 0,
                "use_rand_truncation": False, "min_duration": 2, "insert_white_noise": False,
                "min_white_noise_amp": 0, "max_white_noise_amp": 0}}), path)
            datasets[dataset] = path.name
        model = OmegaConf.create({
            "model": {"model_type": "dynamic_kan" if head == "kan" else "dynamic",
                      "model_name": "tiny-local", "mlp_input_dim": 8, "mlp_hidden_dim": 12,
                      "mlp_num_layers": 1, "mlp_output_size": 1, "mlp_dropout": 0,
                      "mlp_activation_func": "relu", "num_feature_layers": 2, "last_layer": True,
                      "layer_weight_strategy": "per_layer", "pooling_strategy": "mean",
                      "relukan_k": 2, "relukan_grid": 5},
            "train": {"batch_size": 4, "shuffle": False, "num_workers": 0},
            "optimizer": {"name": "adamw", "params": {"learning_rate": 0.00001,
                          "min_learning_rate": 0.0000001, "eps": 1e-8,
                          "weight_decay": 1e-6, "betas": [0.9, 0.98]}},
            "scheduler": {"name": "CosineWarmupLR", "params": {"warmup_steps": 0}},
            "trainer": {"max_epochs": 1, "num_sanity_val_steps": 0, "enable_progress_bar": False,
                        "enable_model_summary": False, "log_every_n_steps": 1},
            "model_checkpoint": {"monitor": "val/spearman", "mode": "max"},
            "early_stopping": {"patience": 2}, "tags": [],
        })
        model_path = root / f"{head}.yaml"
        OmegaConf.save(model, model_path)
        protocol_path = root / "protocol.yaml"
        OmegaConf.save(OmegaConf.create({"configs": [model_path.name], "datasets": datasets,
                                        "seed": 42, "save_head_inputs": True}), protocol_path)
        return sequence.load_plan(protocol_path)

    def test_full_chain_real_mlp_and_kan_reload_and_resume(self):
        torch.set_num_threads(2)
        for head in ("kan", "mlp"):
            with self.subTest(head=head), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                protocol, plans = self.fixture(root, head)
                sequence.preflight(plans)
                configs = plans[0][2]
                if head == "mlp":
                    for config in configs:
                        config.scheduler = None  # no implicit AdamW default LR
                # Audio varies per sample to make Spearman well-defined.
                def load_audio(dataset, filepath):
                    dataset._counter = getattr(dataset, "_counter", 0) + 1
                    value = dataset._counter % 4 + 1
                    return (torch.linspace(-1, 1, 32).mul(value).add(value / 3)[None], 16000)
                with patch("models.base_models.AutoConfig.from_pretrained", return_value=SimpleNamespace()), \
                     patch("models.base_models.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()), \
                     patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=processor), \
                     patch("utils.dataloader.DynamicDataset._load_wav", new=load_audio):
                    stages = sequence.run_experiment(root / "output", configs, protocol, "cpu")
                    self.assertEqual([s["dataset"] for s in stages], list(sequence.SEQUENCE))
                    self.assertEqual(sum(len(s["evaluations"]) for s in stages), 14)
                    self.assertIsNone(stages[0]["initialized_from"])
                    for before, after in zip(stages, stages[1:]):
                        self.assertEqual(before["best_checkpoint"], after["initialized_from"])
                    for stage in stages:
                        attempt = Path(stage["attempt"])
                        with (attempt / "head_inputs.csv").open(encoding="utf-8-sig") as stream:
                            vectors = list(csv.DictReader(stream, delimiter=";"))
                        self.assertEqual(len(vectors), 32)
                        active = DATASET_IDS[stage["dataset"]]
                        # Zero padding for both heads. KAN active block: sigmoid(LayerNorm) in (0, 1).
                        padding = 0.0
                        values = [(int(r["bloco"]), float(r["valor"].replace(",", "."))) for r in vectors]
                        self.assertTrue(all(value == padding for block, value in values if block != active))
                        if head == "kan":
                            self.assertTrue(all(0 < value < 1 for block, value in values if block == active))
                        with (attempt / "epochs.csv").open(encoding="utf-8-sig") as stream:
                            epoch = next(csv.DictReader(stream, delimiter=";"))
                        self.assertNotEqual(epoch["pred_min"], "")
                        if head == "mlp":
                            self.assertEqual(float(epoch["lr"].replace(",", ".")), 0.00001)
                    # Frozen backbone remains identical across the five checkpoints.
                    initial_weights = torch.load(stages[0]["best_checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
                    final_weights = torch.load(stages[-1]["best_checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
                    for key in initial_weights:
                        if key.startswith("model.backbone."):
                            torch.testing.assert_close(initial_weights[key], final_weights[key], rtol=0, atol=0)
                    rows = result_rows(stages, configs[0])
                    last_bvcc = next(r for r in rows if r["etapa"] == 5 and r["dataset_teste"] == "bvcc")
                    self.assertEqual(last_bvcc["etapa_referencia"], 1)
                    self.assertAlmostEqual(last_bvcc["esquecimento_mse"],
                                           stages[4]["evaluations"]["bvcc"]["mse"] - stages[0]["evaluations"]["bvcc"]["mse"])
                    with patch.object(sequence, "train_stage", side_effect=AssertionError("must not retrain")), \
                         patch.object(sequence, "evaluate", side_effect=AssertionError("must not reevaluate")):
                        sequence.run_experiment(root / "output", configs, protocol, "cpu", resume=True)
                    # A changed config must not silently reuse results.
                    changed = [OmegaConf.create(OmegaConf.to_container(c)) for c in configs]
                    changed[0].optimizer.params.learning_rate = 0.00002
                    with self.assertRaisesRegex(ValueError, "changed"):
                        sequence.run_experiment(root / "output", changed, protocol, "cpu", resume=True)

    def test_config_arbitrary_width_and_protocol_overrides_old_compact_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, plans = self.fixture(root)
            model_path = plans[0][1]
            config = OmegaConf.load(model_path)
            config.model.mlp_input_dim = 768
            config.model.dataset_block_expansion = {"enabled": True, "output_multiplier": 2.5,
                                                    "padding_value": 0.1, "block_width": 1024}
            OmegaConf.save(config, model_path)
            _, plans = sequence.load_plan(root / "protocol.yaml")
            for config in plans[0][2]:
                self.assertEqual(config.model.mlp_input_dim, 768)
                self.assertEqual(config.model.dataset_block_expansion.output_multiplier, 4)
                self.assertNotIn("block_width", config.model.dataset_block_expansion)
                self.assertFalse(config.model.layer_norm_affine)
                # Old fill 0.1 replaced by zero padding (same for KAN and MLP).
                self.assertEqual(config.model.dataset_block_expansion.padding_value, 0.0)

    def test_mlp_protocol_keeps_zero_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, plans = self.fixture(root, head="mlp")
            for config in plans[0][2]:
                self.assertEqual(config.model.model_type, "dynamic")
                self.assertEqual(config.model.dataset_block_expansion.padding_value, 0.0)

    def test_resume_after_partial_evaluation_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, plans = self.fixture(root)
            configs = plans[0][2]
            counter = []
            def fake_train(config, stage_dir, previous, gpu, protocol):
                best = stage_dir / "best.ckpt"
                best.touch()
                counter.append(str(best))
                record = {"best_checkpoint": str(best), "initialized_from": previous}
                sequence.json_write(stage_dir / "trained.json", record)
                return record
            def fake_evaluate(config, checkpoint, output_dir, gpu, batch_size):
                result = {"srcc": 0.7, "mse": 0.4, "num_examples": 4,
                          "srcc_undefined": False, "constant_predictions": False,
                          "checkpoint": checkpoint, "test_metadata_sha256": sequence.hashlib.sha256(
                              Path(config.datasets.test[0].metadata_path).read_bytes()).hexdigest()}
                sequence.json_write(output_dir / "metrics.json", result)
                return result
            with patch.object(sequence, "train_stage", side_effect=fake_train), \
                 patch.object(sequence, "evaluate", side_effect=RuntimeError("temporary test failure")):
                with self.assertRaises(RuntimeError):
                    sequence.run_experiment(root / "output", configs, protocol, "cpu")
            with patch.object(sequence, "train_stage", side_effect=fake_train), \
                 patch.object(sequence, "evaluate", side_effect=fake_evaluate):
                sequence.run_experiment(root / "output", configs, protocol, "cpu", resume=True)
            self.assertEqual(len(counter), 5)  # first stage reused; not trained twice

    def test_csv_decimal_comma_preserves_filenames_and_unicode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "values.csv"
            csv_write(path, [{"arquivo": "áudio.v1.wav", "valor": 0.125}], ["arquivo", "valor"])
            csv_write(path, [{"arquivo": "b.wav", "valor": 1.5}], ["arquivo", "valor"], append=True)
            text = path.read_text(encoding="utf-8-sig")
            self.assertEqual(text.count("arquivo;valor"), 1)
            self.assertIn("áudio.v1.wav;0,125", text)
            self.assertIn("b.wav;1,5", text)
            self.assertNotIn("\ufeff", text)

    def test_independent_config_continues_after_failure_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, plans = self.fixture(root)
            plans.append(("02-second", plans[0][1], plans[0][2]))
            with patch.object(sequence, "load_plan", return_value=(protocol, plans)), \
                 patch.object(sequence, "preflight"), \
                 patch.object(sequence, "run_experiment", side_effect=[RuntimeError("first failed"), []]) as run, \
                 patch.object(sys, "argv", ["train_sequence.py", "--output", str(root / "output"), "--gpu", "cpu"]):
                with self.assertRaises(SystemExit) as raised:
                    sequence.main()
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(run.call_count, 2)
            self.assertFalse((root / "output/.sequence.lock").exists())
            self.assertTrue((root / "output/results.csv").is_file())

    def test_constant_predictions_are_flagged_not_converted_to_zero_srcc(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, features):
                return torch.zeros(features[2].shape[0])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, plans = self.fixture(root)
            config = plans[0][2][0]
            with patch("models.calmos_wrapper.CALMOSWrapper.load_from_checkpoint", return_value=ConstantModel()), \
                 patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=processor), \
                 patch("utils.dataloader.DynamicDataset._load_wav", return_value=(torch.ones(1, 32), 16000)):
                result = sequence.evaluate(config, "fake.ckpt", root / "evaluation", "cpu", 4)
            self.assertIsNone(result["srcc"])
            self.assertTrue(result["srcc_undefined"])
            self.assertTrue(result["constant_predictions"])
            self.assertAlmostEqual(result["mse"], 7.5)
            self.assertTrue((root / "evaluation/predictions.csv").is_file())

    def test_validation_uses_validation_columns_and_directory(self):
        from utils.utils import build_dataloaders
        with tempfile.TemporaryDirectory() as temporary:
            _, plans = self.fixture(Path(temporary))
            config = plans[0][2][0]
            config.datasets.val[0].filename_column = "validation_filename"
            config.datasets.val[0].target_column = "validation_mos"
            config.datasets.val[0].base_dir = "/different-validation-directory"
            with patch("utils.utils.DynamicDataset") as dataset:
                build_dataloaders(config)
            validation = dataset.call_args_list[1].kwargs
            self.assertEqual(validation["filename_column"], "validation_filename")
            self.assertEqual(validation["target_column"], "validation_mos")
            self.assertEqual(validation["base_dir"], "/different-validation-directory")

    def test_wandb_offline_real_native_charts_and_full_debug_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, plans = self.fixture(root)
            protocol.wandb = {"enabled": True, "offline": True, "project": "calmos-debug-tests"}
            config = plans[0][2][0]
            def load_audio(dataset, filepath):
                dataset._counter = getattr(dataset, "_counter", 0) + 1
                value = dataset._counter % 4 + 1
                return (torch.linspace(-1, 1, 32).mul(value).add(value / 3)[None], 16000)
            with patch("models.base_models.AutoConfig.from_pretrained", return_value=SimpleNamespace()), \
                 patch("models.base_models.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()), \
                 patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=processor), \
                 patch("utils.dataloader.DynamicDataset._load_wav", new=load_audio):
                result = sequence.train_stage(config, root / "offline-stage", None, "cpu", protocol)
            attempt = Path(result["attempt"])
            self.assertTrue((attempt / "debug_epoch_001.png").is_file())
            snapshot = np.load(attempt / "debug_epoch_001.npz", allow_pickle=False)
            self.assertEqual(snapshot["head_input"].shape, (32,))
            self.assertEqual(snapshot["validation_predictions"].shape, (4,))
            details = json.loads((attempt / "debug_epoch_001.json").read_text())
            self.assertEqual(set(details["kan_support"]), {"0", "1"})
            # sigmoid(LayerNorm) keeps every real input inside the grid; padding is excluded.
            self.assertEqual(details["kan_support"]["0"]["fora_de_todas_bases"], 0.0)
            self.assertGreater(details["kan_support"]["0"]["entrada_min"], 0.0)
            self.assertTrue(details["gradient_norms"])
            if os.environ.get("CALMOS_DEBUG_PREVIEW_DIR"):
                destination = Path(os.environ["CALMOS_DEBUG_PREVIEW_DIR"])
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(attempt / "debug_epoch_001.png", destination / "wandb_debug_preview.png")
                shutil.copy2(attempt / "debug_epoch_001.json", destination / "wandb_debug_preview.json")


if __name__ == "__main__":
    unittest.main()
