import numpy as np
from probpose.heatmap import get_heatmap_expected_value
from numpy.testing import assert_allclose

def test_get_heatmap_expected_value_backends():
    heatmaps = np.random.rand(20, 256, 256).astype(np.float32)
    sigmas = np.random.rand(20).astype(np.float32)

    _, _, scipy_result = get_heatmap_expected_value(heatmaps, sigmas, return_heatmap = True, backend="scipy")
    _, _, torch_result = get_heatmap_expected_value(heatmaps, sigmas, return_heatmap = True, backend="torch")
    assert scipy_result.shape == torch_result.shape
    assert_allclose(scipy_result, torch_result, rtol=1e-5, atol=1e-8)