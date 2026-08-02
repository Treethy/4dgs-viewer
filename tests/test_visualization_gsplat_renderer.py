import numpy as np
import pytest
import torch

from viewer4d import GaussianFrame
from viewer4d.visualization.camera import RenderCamera
from viewer4d.visualization.gsplat_renderer import GsplatRenderer, infer_sh_degree


def test_infer_sh_degree():
    assert infer_sh_degree(1) == 0
    assert infer_sh_degree(16) == 3
    with pytest.raises(ValueError, match="perfect square"):
        infer_sh_degree(15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_renderer_uses_expected_gsplat_arguments():
    device = torch.device("cuda")
    frame = GaussianFrame(
        means=torch.zeros(1, 3, device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        scales=torch.ones(1, 3, device=device),
        opacities=torch.ones(1, device=device),
        sh_coeffs=torch.zeros(1, 16, 3, device=device),
    )
    camera = RenderCamera(
        viewmat=torch.eye(4, device=device),
        K=torch.eye(3, device=device),
        width=4,
        height=3,
        near=0.01,
        far=100.0,
    )
    captured = {}

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        image = torch.full((1, 3, 4, 3), 0.25, device=device)
        alpha = torch.ones((1, 3, 4, 1), device=device)
        return image, alpha, {}

    renderer = GsplatRenderer(rasterization_fn=fake_rasterization)
    image = renderer.render(frame, camera)
    assert image.shape == (3, 4, 3)
    assert image.dtype == np.uint8
    assert captured["sh_degree"] == 3
    assert captured["width"] == 4
    assert captured["height"] == 3
