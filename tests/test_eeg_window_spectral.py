import numpy as np
import torch

from temporal_ssl.data import Normalizer
from temporal_ssl.eeg_window_spectral import band_targets, extract_windows, bootstrap_differences


def test_known_frequency_targets_channel_weight_and_scale():
    t = np.arange(2000)/500
    # Equal channel weights despite very different channel amplitudes.
    window = np.stack([np.sin(2*np.pi*2*t), 100*np.sin(2*np.pi*10*t)], -1)
    targets = band_targets(np.stack([window, window*17]))
    np.testing.assert_allclose(targets[:, [0, 2]], .5, atol=1e-8)
    np.testing.assert_allclose(targets[:, [1, 3, 4]], 0, atol=1e-8)
    np.testing.assert_allclose(targets.sum(1), 1)
    # Gamma is actually measured, rather than accidentally omitted from denominator.
    gamma = np.sin(2*np.pi*35*t)[None, :, None]
    np.testing.assert_allclose(band_targets(gamma), [[0, 0, 0, 0, 1]], atol=1e-8)


def test_window_extraction_keeps_rows_and_excludes_future():
    class Model:
        def representations(self, x):
            assert x.shape[1] == 2000
            return {'local': x.reshape(len(x), 80, 25, 1).mean(2)}
    class Dataset:
        normalizer = Normalizer(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        x = np.zeros((3, 4000, 1), dtype=np.float32)
        def __len__(self):
            return len(self.x)
    data = Dataset()
    data.x[:, :2000] = np.arange(3)[:, None, None]
    data.x[:, 2000:] = 10000
    out = extract_windows(Model(), data, 'cpu')
    np.testing.assert_array_equal(out['mean'][:, 0], [0, 1, 2])
    assert out['mean_std'].shape == (3, 2)
    np.testing.assert_array_equal(out['mean_std'][:, 1], 0)


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    rng = np.random.default_rng(12)
    truth = rng.normal(size=(12, 5))
    pred = rng.normal(size=(12, 5))
    ci = bootstrap_differences(truth, pred, pred, np.repeat(['a', 'b', 'c'], 4), draws=20)
    np.testing.assert_array_equal(ci, np.zeros((2, 6)))
