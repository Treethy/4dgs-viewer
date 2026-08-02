import numpy as np
import torch

from viewer4d import GaussianFrame
from viewer4d.visualization.viser_adapter import gaussian_frame_to_viser


def test_gaussian_frame_to_viser_identity_covariance_and_dc_color():
    frame = GaussianFrame(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.tensor([[2.0, 3.0, 4.0]]),
        opacities=torch.tensor([0.75]),
        sh_coeffs=torch.zeros(1, 16, 3),
    )
    data = gaussian_frame_to_viser(frame)
    assert data.centers.shape == (1, 3)
    assert np.allclose(data.covariances[0], np.diag([4.0, 9.0, 16.0]))
    assert np.allclose(data.rgbs[0], np.array([0.5, 0.5, 0.5]))
    assert np.allclose(data.opacities, np.array([[0.75]], dtype=np.float32))
