"""Small CPU regressions; no external data or downloaded model weights."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import torch

from eva1.config import DatasetConfig, load_config
from eva1.data import HSIFolder, read_cube
from eva1.engine import meta_step, train_step
from eva1.losses import loss_grad
from eva1.models import LPN, Reconstruction
from eva1.runtime import classification_metrics

torch.set_num_threads(2)


def small_config(**kwargs):
    return DatasetConfig(**{"channels": 3, "num_classes": 2, "rgb_bands": [0, 1, 2],
                            "layout": "CHW", "normalization": "scale", "scale": 1000, **kwargs})


class DataTests(unittest.TestCase):
    def test_presets(self):
        for name, channels in [("HSC-XMS", 150), ("HSC-ZHS", 32), ("HSRS-SC", 48), ("OHS", 32)]:
            self.assertEqual(load_config(name).channels, channels)

    def test_layout_normalization_and_band_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tif"
            cube = np.stack([np.full((8, 8), v, dtype=np.float32) for v in [0, 500, 2000]])
            tifffile.imwrite(path, cube, photometric="minisblack")
            chw = read_cube(path, small_config())
            torch.testing.assert_close(chw[:, 0, 0], torch.tensor([0.0, 0.5, 1.0]))
            tifffile.imwrite(path, cube.transpose(1, 2, 0), photometric="rgb")
            torch.testing.assert_close(read_cube(path, small_config(layout="HWC")), chw)
            reordered = read_cube(path, small_config(layout="HWC", band_indices=[2, 0, 1]))
            torch.testing.assert_close(reordered, chw[[2, 0, 1]])

    def test_constant_and_zero_bands_are_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tif"
            cube = np.stack([np.zeros((8, 8)), np.ones((8, 8)), np.arange(64).reshape(8, 8)]).astype("float32")
            tifffile.imwrite(path, cube, photometric="minisblack")
            output = read_cube(path, small_config(normalization="band_minmax"))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output[0].sum().item(), 0)
            self.assertEqual(output[1].min().item(), 1)
            self.assertEqual(output[2].max().item(), 1)

    def test_eval_preserves_training_labels_when_a_class_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b").mkdir()
            tifffile.imwrite(root / "b" / "one.tif", np.ones((3, 8, 8), dtype="float32"), photometric="minisblack")
            dataset = HSIFolder(root, small_config(), classes=["a", "b"])
            self.assertEqual(dataset[0][1], 1)
            with self.assertRaises(ValueError):
                HSIFolder(root, small_config())
            (root / "unknown").mkdir()
            with self.assertRaises(ValueError):
                HSIFolder(root, small_config(), classes=["a", "b"])

    def test_wrong_channel_count_is_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tif"
            tifffile.imwrite(path, np.ones((4, 8, 8), dtype="float32"), photometric="minisblack")
            with self.assertRaisesRegex(ValueError, "band_indices"):
                read_cube(path, small_config())


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.reconstruction = Reconstruction(inc=3, base_ch=4, rgb_band=[0, 1, 2])
        self.lpn = LPN(inc=3, dim=8)
        self.classifier = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.AdaptiveAvgPool2d(1),
                                               torch.nn.Flatten(), torch.nn.Linear(4, 2))
        self.optimizer_r = torch.optim.Adam(self.reconstruction.parameters(), lr=1e-3)
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        self.batch = (torch.rand(2, 3, 8, 8), torch.tensor([0, 1]))
        self.coefficients = (1.0, 0.2, 0.1)

    def test_output_shapes_and_ranges(self):
        rgb, spectral = self.reconstruction(self.batch[0])
        self.assertEqual(rgb.shape, self.batch[0].shape)
        self.assertEqual(spectral.shape, self.batch[0].shape)
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        self.assertEqual(self.lpn(self.batch[0]).shape, (2, 1, 8, 8))
        with torch.no_grad():
            fast_rgb, _ = self.reconstruction(self.batch[0], meta=True)
        torch.testing.assert_close(fast_rgb, rgb)

    def test_training_and_second_order_meta_update(self):
        before_r = deepcopy(self.reconstruction.state_dict())
        before_lpn = deepcopy(self.lpn.state_dict())
        losses = train_step(self.reconstruction, self.classifier, self.lpn, self.optimizer_r,
                            self.optimizer_c, self.batch, [0, 1, 2], self.coefficients)
        self.assertTrue(all(np.isfinite(losses)))
        self.assertTrue(any(not torch.equal(value, before_r[key]) for key, value in self.reconstruction.state_dict().items()))
        for key, value in self.lpn.state_dict().items():
            torch.testing.assert_close(value, before_lpn[key], rtol=0, atol=0)
        before_c = deepcopy(self.classifier.state_dict())
        before_r = deepcopy(self.reconstruction.state_dict())
        before_optimizer = deepcopy(self.optimizer_c.state_dict())
        query = (torch.rand_like(self.batch[0]), torch.tensor([1, 0]))
        losses = meta_step(self.reconstruction, self.classifier, self.lpn, self.optimizer_c,
                           self.batch, query, [0, 1, 2], self.coefficients, 1e-3)
        self.assertTrue(all(np.isfinite(losses)))
        for model, previous in [(self.classifier, before_c), (self.reconstruction, before_r)]:
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, previous[key], rtol=0, atol=0)
        after_optimizer = self.optimizer_c.state_dict()
        self.assertEqual(after_optimizer["param_groups"], before_optimizer["param_groups"])
        for key, state in before_optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(after_optimizer["state"][key][name], value, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before_lpn[key]) for key, value in self.lpn.state_dict().items()))
        for module in self.reconstruction.modules():
            if hasattr(module, "named_leaves"):
                for name, _ in module.named_leaves():
                    self.assertIsNone(getattr(module, name + "_meta"))

    def test_gradient_loss_matches_original_formula(self):
        rgb, reference = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
        total = 0
        for values in [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]:
            kernel = torch.tensor(values, dtype=rgb.dtype)[None, None].repeat(1, 3, 1, 1)
            output = torch.nn.functional.conv2d(rgb, kernel, padding=1)
            target = torch.nn.functional.conv2d(reference, kernel, padding=1)
            total += torch.nn.functional.l1_loss(output, target.abs() * target)
        torch.testing.assert_close(loss_grad(rgb, reference), total)


class MetricTests(unittest.TestCase):
    def test_confusion_matrix_counts_each_sample_once(self):
        result = classification_metrics([0, 0, 1], [0, 1, 1], ["a", "b"])
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["kappa"], 0.4)
        with self.assertRaises(ValueError):
            classification_metrics([0], [0, 0], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
