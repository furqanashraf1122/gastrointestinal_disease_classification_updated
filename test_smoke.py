"""Lightweight regression checks; no images or pretrained downloads required."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import gi_ensemble as gi
import run_repeated_experiments as repeated


class SmokeTests(unittest.TestCase):
    def test_paper_configuration(self):
        cfg = gi.config(Path(__file__).with_name('config.json'))
        self.assertEqual(len(cfg['backbones']), 6)
        self.assertEqual(cfg['fusion']['feature_counts'], [128, 256, 384, 512])
        self.assertAlmostEqual(sum(cfg['fusion']['reported_weights'].values()), 1)
        self.assertEqual([s['feature_dim'] for s in gi.MODEL_SPECS.values()],
                         [1024, 2048, 2048, 2048, 1920, 512])

    def test_joint_validation_weight_search(self):
        y = np.array([0, 1, 0, 1])
        correct = np.eye(2)[y] * .9 + .05
        wrong = correct[:, ::-1]
        weights, score = gi.select_weights(np.stack([wrong, correct]), y, .1)
        self.assertEqual(score, 1)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue(np.all(weights >= 0))
        with self.assertRaises(ValueError):
            gi.select_weights(np.stack([wrong, correct]), y, .3)

    def test_table10_fusion_generation(self):
        rng = np.random.default_rng(17)
        labels = np.tile(np.arange(3), 20)
        def make_features(y):
            return [np.eye(3)[y] + rng.normal(0, .05, (len(y), 3)) for _ in range(2)]
        projected = {
            "train": make_features(labels),
            "val": make_features(labels),
            "test": make_features(labels),
        }
        splits = {name: {"y": labels} for name in ("train", "val", "test")}
        result = gi.evaluate_feature_fusion(projected, splits, [2, 3, 4])
        self.assertEqual(set(result), {"feature_averaging", "feature_concatenation"})
        for row in result.values():
            self.assertEqual(row["selected_features"], 2)
            self.assertGreaterEqual(row["accuracy"], 0.95)

    def test_repeated_run_summary_uses_sample_sd(self):
        with tempfile.TemporaryDirectory() as d:
            run_dirs = []
            for seed, acc in ((42, .8), (43, 1.0), (44, .9)):
                folder = Path(d) / f"seed_{seed}"
                folder.mkdir()
                (folder / "metrics.json").write_text(json.dumps({
                    "run_seed": seed,
                    "A5": {"accuracy": acc, "weights": {"a": .5, "b": .5}},
                }))
                run_dirs.append(folder)
            _, summary = repeated.aggregate_metric_files(
                run_dirs, "metrics.json", ["A5"])
            self.assertAlmostEqual(summary["A5"]["accuracy"]["mean"], .9)
            self.assertAlmostEqual(summary["A5"]["accuracy"]["sample_sd"], .1)

    def test_feature_alignment_rejects_different_order(self):
        with tempfile.TemporaryDirectory() as d:
            for split in ('train', 'val', 'test'):
                for name in ('a', 'b'):
                    np.savez(Path(d) / f'{name}_{split}.npz',
                             features=np.zeros((2, 4)), probabilities=np.eye(2),
                             y=np.array([0, 1]), paths=np.array(['a', 'b'] if name == 'a' else ['b', 'a']))
            with self.assertRaisesRegex(ValueError, 'do not align'):
                gi.load_features(d, ['a', 'b'])

    def test_extraction_does_not_resample_training_rows(self):
        data = types.ModuleType('torch.utils.data')
        data.Dataset = object
        data.DataLoader = lambda dataset, **kwargs: types.SimpleNamespace(dataset=dataset, **kwargs)
        data.WeightedRandomSampler = lambda weights, **kwargs: ('weighted', weights)
        df = pd.DataFrame({'path': ['a', 'b'], 'label': ['x', 'y'], 'split': ['train', 'train']})
        with patch.dict(sys.modules, {'torch.utils.data': data}), patch.object(gi, 'transforms_for', return_value=None):
            extract = gi.loader(df, '.', {'x': 0, 'y': 1}, 224, 'train', 2, 0)
            train = gi.loader(df, '.', {'x': 0, 'y': 1}, 224, 'train', 2, 0, balance_training=True)
            self.assertIsNone(extract.sampler)
            self.assertFalse(extract.shuffle)
            self.assertIsNotNone(train.sampler)

    def test_pretrained_auxiliary_heads_disabled_after_loading(self):
        torch = types.ModuleType('torch')
        torch.nn = types.SimpleNamespace(Module=object, Linear=lambda a,b: (a,b), Identity=lambda: 'identity')
        vision = types.ModuleType('torchvision')
        calls = []
        def constructor(dim):
            def create(**kw):
                calls.append(kw)
                # Mimic torchvision's pretrained auxiliary-head constraint.
                if kw['weights'] is not None and not kw['aux_logits']:
                    raise ValueError('Pretrained models require aux_logits=True at construction')
                return types.SimpleNamespace(fc=types.SimpleNamespace(in_features=dim), aux_logits=True,
                                             aux1='aux', aux2='aux', AuxLogits='aux')
            return create
        vision.models = types.SimpleNamespace(googlenet=constructor(1024), inception_v3=constructor(2048))
        with patch.dict(sys.modules, {'torch': torch, 'torchvision': vision}):
            for name in ('googlenet', 'inception_v3'):
                model = gi.backbone(name, 8, pretrained=True)
                self.assertFalse(model.features.aux_logits)
                self.assertFalse(calls[-1]['transform_input'])
                if name == 'googlenet':
                    self.assertIsNone(model.features.aux1)
                    self.assertIsNone(model.features.aux2)
                else:
                    self.assertIsNone(model.features.AuxLogits)


if __name__ == '__main__':
    unittest.main()
