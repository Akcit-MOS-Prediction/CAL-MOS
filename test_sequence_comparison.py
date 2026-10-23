"""CPU checks of mathematical curves, routing, lineage and final comparison gating."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
import torch

from models.torch_relu_kan import ReLUKAN, ReLUKANLayer
from models.base_models import MLPBase
from utils.sequence_comparison import (
    DATASET_IDS, SEQUENCE, ProbeCollector, analyze_experiment, basis_values, build_head,
    generate_report, load_checkpoint, region_change, region_overlap, run_head,
    select_edges, sequence_records, support_rows, weighted_curve, write_json,
    schedule_comparison,
)


def diagnostic_config(kind="kan", grid=5):
    return {"seed": 42, "model": {
        "model_name": "diagnostic/synthetic-backbone", "model_type": "dynamic_kan" if kind == "kan" else "dynamic",
        "mlp_input_dim": 4, "mlp_hidden_dim": 3, "mlp_num_layers": 1, "mlp_output_size": 1,
        "mlp_dropout": 0., "mlp_activation_func": "relu", "use_layer_norm": True,
        "layer_norm_affine": False, "layer_norm_eps": 1e-5, "freeze_backbone": True,
        "last_layer": True, "pooling_strategy": "mean", "relukan_k": 2, "relukan_grid": grid,
        "dataset_block_expansion": {"enabled": True, "num_blocks": 4, "output_multiplier": 4.,
                                     "padding_value": -1. if kind == "kan" else 0.,
                                     "dataset_to_id": DATASET_IDS}},
        "optimizer": {"name": "adamw", "params": {"learning_rate": .0005}},
        "train": {"batch_size": 8}, "trainer": {"max_epochs": 2, "accumulate_grad_batches": 1},
        "early_stopping": {"patience": 2}}


def make_head(config):
    model = config["model"]
    kwargs = {"dataset_block_expansion": model["dataset_block_expansion"], "layer_norm_affine": False}
    if model["model_type"] == "dynamic_kan":
        return ReLUKAN([4, 3, 1], model["relukan_grid"], 2, **kwargs).eval()
    return MLPBase(input_size=4, hidden_dim=3, num_layers=1, output_size=1, dropout=0., **kwargs).eval()


def fixture_experiment(root, kind="kan", grid=5, complete=True, probes=True):
    """Synthetic data for offline verification ONLY; never publish these fixtures to W&B."""
    directory = Path(root) / f"{kind}-g{grid}"
    directory.mkdir(parents=True)
    write_json(directory / "manifest.json", {"sequence": SEQUENCE})
    config = diagnostic_config(kind, grid)
    torch.manual_seed(123)
    head = make_head(config)
    initial = directory / "initial.pt"
    torch.save({"head_state": head.state_dict()}, initial)
    previous = None
    seen = []
    for index, dataset in enumerate(SEQUENCE, 1):
        stage = directory / f"{index:02d}-{dataset}"
        stage.mkdir()
        OmegaConf.save(OmegaConf.create(config), stage / "config.yaml")
        with torch.no_grad():
            final = head.rk_layers[-1].equal_size_conv if kind == "kan" else head.layers[-1]
            final.bias.add_(.01 * index)
            final.weight.add_(.002 * index)
            if kind == "kan":
                head.rk_layers[0].equal_size_conv.weight.mul_(1 + .02 * index)
                head.rk_layers[0].phase_low.add_(.001 * index)
        checkpoint = stage / "best.ckpt"
        torch.save({"state_dict": {f"model.mlp.{key}": value for key, value in head.state_dict().items()}}, checkpoint)
        result = {"best_checkpoint": str(checkpoint), "initialized_from": previous}
        if index == 1:
            result["initial_head_checkpoint"] = str(initial)
        write_json(stage / "trained.json", result)
        if dataset not in seen:
            seen.append(dataset)
        for tested in seen:
            write_json(stage / "evaluations" / tested / "metrics.json", {
                "checkpoint": str(checkpoint), "srcc": .8 - index * .01, "mse": .2 + index * .01,
                "num_examples": 8, "test_metadata_sha256": "synthetic-test-" + tested})
        previous = str(checkpoint)
    write_json(directory / "progress.json", {"status": "finished" if complete else "running", "stages": []})
    if probes:
        rng = np.random.default_rng(42)
        for dataset in DATASET_IDS:
            values = rng.normal(size=(8, 4)).astype(np.float32)
            targets = np.linspace(1, 4, 8).tolist()
            collector = ProbeCollector(head, 8)
            run_head(head, values, dataset)
            collector.close()
            collector.save(directory / "comparison/probes" / f"{dataset}.npz", targets,
                           [f"synthetic-{dataset}-{index}" for index in range(8)],
                           {"dataset": dataset, "split": "test", "backbone": config["model"]["model_name"],
                            "test_metadata_sha256": "synthetic-test-" + dataset})
    return directory


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_weighted_basis_equals_actual_layer(self):
        layer = ReLUKANLayer(1, 5, 2, 1)
        with torch.no_grad():
            layer.equal_size_conv.bias.zero_()
        x = np.linspace(-.6, 1.6, 101).astype(np.float32)
        expected = weighted_curve(x, layer.phase_low[0].detach().numpy(),
                                  layer.phase_height[0].detach().numpy(),
                                  layer.equal_size_conv.weight[0, 0, :, 0].detach().numpy(), layer.r)
        actual = layer(torch.from_numpy(x[:, None])).detach().numpy()[:, 0]
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)

    def test_no_change_has_no_spurious_overlap(self):
        values = np.linspace(-1, 1, 100)
        delta, mask, threshold = region_change(values, values)
        self.assertFalse(mask.any())
        self.assertIsNone(region_overlap(delta, mask, delta, mask)["jaccard_regiao_alterada"])
        self.assertGreater(threshold, 0)

    def test_local_curve_change_and_signed_overlap(self):
        x = np.linspace(0, 1, 101)
        before = weighted_curve(x, np.array([.1]), np.array([.4]), np.array([1.]), 1.)
        after = 2 * before
        delta, mask, _ = region_change(before, after)
        self.assertFalse(mask[x > .4].any())
        stats = region_overlap(delta, mask, -delta, mask)
        self.assertEqual(stats["jaccard_regiao_alterada"], 1.)
        self.assertAlmostEqual(stats["cosseno_delta_assinado"], -1.)

    def test_learned_padding_leakage_not_hidden_as_global_coverage(self):
        head = make_head(diagnostic_config())
        values = np.arange(32, dtype=np.float32).reshape(8, 4) / 10
        _, inputs = run_head(head, values, "bvcc", capture=True)
        rows = support_rows(head, inputs, "01-bvcc", "bvcc")
        padding = next(row for row in rows if row["regiao"] == "padding")
        self.assertEqual(padding["padding_base_maximo"], 0.)
        self.assertEqual(padding["fracao_sem_base_ativa"], 1.)
        with torch.no_grad():
            head.rk_layers[0].phase_low[4:].fill_(-1.1)
            head.rk_layers[0].phase_height[4:].fill_(-.9)
        changed = support_rows(head, inputs, "02-singmos", "bvcc")
        self.assertGreater(next(row for row in changed if row["regiao"] == "padding")["padding_base_maximo"], 0)

    def test_probe_is_raw_and_bounded_and_restore_is_exact(self):
        config = diagnostic_config()
        head = make_head(config)
        raw = np.arange(20, dtype=np.float32).reshape(5, 4)
        collector = ProbeCollector(head, 3)
        expected, _ = run_head(head, raw, "singmos")
        collector.close()
        np.testing.assert_array_equal(np.concatenate(collector.values), raw[:3])
        restored = build_head(config, head.state_dict())
        actual, _ = run_head(restored, raw, "singmos")
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(restored.rk_layers), 2)
        selection = select_edges(restored)
        with torch.no_grad():
            restored.rk_layers[0].equal_size_conv.weight[:, :, :, 5].add_(100.)
        self.assertEqual(selection, select_edges(restored))

    def test_no_final_report_for_failed_or_incomplete_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = fixture_experiment(tmp, complete=False)
            status = generate_report([directory], Path(tmp) / "report")
            self.assertEqual(status["status"], "waiting_for_complete_sequences")
            self.assertFalse(status["wandb_run_created"])
            self.assertFalse((Path(tmp) / "report/dashboard.html").exists())

    def test_lineage_checked_before_comparing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = fixture_experiment(tmp)
            path = directory / "02-singmos/trained.json"
            trained = json.loads(path.read_text())
            trained["initialized_from"] = "/wrong/checkpoint"
            write_json(path, trained)
            with self.assertRaisesRegex(ValueError, "lineage"):
                sequence_records(directory)

    def test_full_kan_mlp_report_and_first_bvcc_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            kan = fixture_experiment(tmp, "kan")
            mlp = fixture_experiment(tmp, "mlp")
            destination = Path(tmp) / "report"
            status = generate_report([kan, mlp], destination,
                                     {"grid_points": 21, "inputs_per_block": 1, "response_points": 9})
            self.assertTrue(status["complete"])
            self.assertTrue((destination / "dashboard.html").is_file())
            self.assertTrue((destination / "overlap.csv").is_file())
            result = analyze_experiment(sequence_records(kan), {
                "probe_samples": 8, "grid_points": 21, "inputs_per_block": 1, "outputs_per_layer": 1,
                "response_points": 9, "conditional_dims": 1, "change_atol": 1e-6, "change_rtol": 1e-3})
            final = next(row for row in result["tables"]["tests"]
                         if row["dataset"] == "bvcc" and row["etapa"] == "05-bvcc")
            self.assertAlmostEqual(final["esquecimento_mse"], .04)
            self.assertGreater(len(result["tables"]["functional"]), 0)
            self.assertTrue(any(row["rmse_mudanca_predicao"] > 0 for row in result["tables"]["functional"]))
            again = generate_report([kan, mlp], destination,
                                    {"grid_points": 21, "inputs_per_block": 1, "response_points": 9})
            self.assertEqual(again["fingerprint"], status["fingerprint"])

    def test_missing_probes_stays_explicit_and_partial_is_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = fixture_experiment(tmp, complete=False, probes=False)
            status = generate_report([directory], Path(tmp) / "partial", {
                "grid_points": 21, "inputs_per_block": 1, "response_points": 9}, allow_partial=True)
            self.assertFalse(status["complete"])
            manifest = json.loads((Path(tmp) / "partial/manifest.json").read_text())
            self.assertEqual(len(manifest["experiments"][0]["warnings"]), 4)
            self.assertEqual(manifest["row_counts"]["functional"], 0)

    def test_queue_waits_for_all_configs_and_schedules_only_one_cpu_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_output, second_output = root / "task1/outputs", root / "task2/outputs"
            first = fixture_experiment(first_output)
            second = fixture_experiment(second_output, "mlp", complete=False)
            write_json(root / "round.json", {"tasks": [
                {"label": "kan", "output": str(first_output)}, {"label": "mlp", "output": str(second_output)}]})
            protocol = {"comparison": {"enabled": True}, "wandb": {"enabled": False}}
            with mock.patch("subprocess.Popen") as popen:
                waiting = schedule_comparison(first_output, [first], protocol)
                self.assertEqual(waiting["status"], "waiting_for_complete_sequences")
                popen.assert_not_called()
                write_json(second / "progress.json", {"status": "finished"})
                popen.return_value.pid = 12345
                scheduled = schedule_comparison(first_output, [first], protocol)
                self.assertEqual(scheduled["status"], "scheduled")
                self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                again = schedule_comparison(first_output, [first], protocol)
                self.assertEqual(again["status"], "analysis_already_scheduled")
                self.assertEqual(popen.call_count, 1)
                (root / "comparison/.scheduled.lock").unlink()
                write_json(root / "comparison/status.json", {"status": "finished", "fingerprint": "saved"})
                schedule_comparison(first_output, [first], protocol)
                status = json.loads((root / "comparison/status.json").read_text())
                self.assertEqual(status["fingerprint"], "saved")


if __name__ == "__main__":
    unittest.main()
