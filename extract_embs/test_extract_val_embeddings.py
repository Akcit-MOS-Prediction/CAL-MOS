"""Synthetic regression tests: no model downloads or production queue changes."""
import csv
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import extract_val_embeddings as extractor


class ValidationExtractionTests(unittest.TestCase):
    def test_default_cache_load_uses_local_snapshot(self):
        args = SimpleNamespace(fallback_cache_dir=Path('/writable'), local_files_only=False)
        factory = Mock()
        with patch('huggingface_hub.snapshot_download', return_value='/readonly/snapshot') as download:
            extractor.load_pretrained(factory, 'test/model', args)
        download.assert_called_once_with('test/model', local_files_only=True)
        factory.from_pretrained.assert_called_once_with('/readonly/snapshot', local_files_only=True)

    def test_missing_model_downloads_only_one_weight_format_to_fallback(self):
        args = SimpleNamespace(fallback_cache_dir=Path('/writable'), local_files_only=False)
        factory = Mock()
        with patch('huggingface_hub.snapshot_download',
                   side_effect=[OSError('missing'), OSError('missing'), '/writable/snapshot']) as download:
            with patch('huggingface_hub.list_repo_files', return_value=['config.json', 'model.safetensors', 'pytorch_model.bin']):
                extractor.load_pretrained(factory, 'test/model', args)
        self.assertEqual(download.call_args.kwargs['allow_patterns'], ['*.json', '*.safetensors'])
        self.assertEqual(download.call_args.kwargs['cache_dir'], '/writable')
        factory.from_pretrained.assert_called_once_with('/writable/snapshot', local_files_only=True)

    def test_validation_only_and_backbone_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config/models").mkdir(parents=True)
            for name in ("encoder", "encoder_kan"):
                (root / f"config/models/{name}.yaml").write_text(yaml.safe_dump({"model": {
                    "model_name": "test/encoder", "layer_norm_eps": 1e-5}}))
            models = extractor.discover(root, None)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["name"], "encoder")
            self.assertEqual(extractor.discover(root, ["encoder_kan"]), models)
            (root / "config/datasets").mkdir()
            metadata = root / "val.csv"
            metadata.write_text("filepath,mos\na.wav,3.0\nb.wav,4.0\n")
            config = {"datasets": {
                "train": [{"metadata_path": "/never/read/train.csv"}],
                "test": [{"metadata_path": "/never/read/test.csv"}],
                "val": [{"metadata_path": str(metadata), "base_dir": str(root),
                         "filename_column": "filepath", "target_column": "mos"}]}}
            (root / "config/datasets/bvcc.yaml").write_text(yaml.safe_dump(config))
            rows, _ = extractor.validation_rows(root, "bvcc", 0)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["audio"], str(root / "a.wav"))

    def test_whisper_padding_excluded_and_segments_weighted_by_frames(self):
        # One full 640-sample chunk and a 320-sample remainder: 2+1 valid frames.
        class Processor:
            def __call__(self, audio, **kwargs):
                return {"input_features": torch.tensor(audio)}

        class Model:
            calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    values = [[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]]
                else:
                    values = [[5.0, 6.0], [1000.0, 1000.0], [1000.0, 1000.0]]
                hidden = torch.tensor([values])
                return SimpleNamespace(hidden_states=(torch.full_like(hidden, -99), hidden))

        pooled = extractor.pooled_last_layer(Model(), Processor(), torch.ones(960), 640,
                                            torch.device("cpu"), 1.0, True)
        torch.testing.assert_close(pooled, torch.tensor([3.0, 4.0]))

    def test_explicit_affine_parameters_and_identity_equivalence(self):
        item = {"name": "encoder", "id": "test/encoder"}
        raw = torch.tensor([1.0, 2.0, 4.0])
        weight, bias, source = extractor.affine_state(item, {}, 3)
        self.assertEqual(source["kind"], "identity")
        normalized = torch.nn.functional.layer_norm(raw, (3,))
        torch.testing.assert_close(normalized, torch.nn.functional.layer_norm(raw, (3,), weight, bias))
        weight, bias, _ = extractor.affine_state(item, {"encoder": {"weight": [2, 2, 2], "bias": [1, 1, 1]}}, 3)
        torch.testing.assert_close(torch.nn.functional.layer_norm(raw, (3,), weight, bias), normalized * 2 + 1)

    def test_pair_table_has_96_unique_undirected_pairs_for_eight_models(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), datasets=list(extractor.DATASETS))
            rows = extractor.pair_rows(args, [{"name": f"model{i}"} for i in range(8)])
            self.assertEqual(len(rows), 96)
            keys = {(row["modelo"], row["layernorm"], row["dataset_a"], row["dataset_b"]) for row in rows}
            self.assertEqual(len(keys), 96)
            self.assertTrue(all(row["dataset_a"] != row["dataset_b"] for row in rows))


if __name__ == "__main__":
    unittest.main()
