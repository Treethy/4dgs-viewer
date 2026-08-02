import math

import numpy as np
import pytest
import torch

from viewer4d.visualization.camera import (
    CameraState,
    camera_state_to_render_camera,
    fit_render_size,
    pinhole_intrinsics,
    quaternion_wxyz_to_matrix,
)


def test_identity_camera_to_world_becomes_expected_world_to_camera():
    state = CameraState(
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([1.0, 2.0, 3.0]),
        fov=math.pi / 2,
        width=800,
        height=600,
        near=0.01,
        far=100.0,
    )
    camera = camera_state_to_render_camera(state, device="cpu", max_width=400)
    assert (camera.width, camera.height) == (400, 300)
    assert torch.allclose(camera.viewmat[:3, 3], torch.tensor([-1.0, -2.0, -3.0]))
    assert camera.K[0, 0].item() == pytest.approx(150.0)
    assert camera.K[1, 1].item() == pytest.approx(150.0)


def test_camera_helpers():
    assert np.allclose(
        quaternion_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0])),
        np.eye(3),
    )
    K = pinhole_intrinsics(width=640, height=480, vertical_fov=math.pi / 2)
    assert K[0, 2] == 320
    assert K[1, 2] == 240
    assert fit_render_size(1920, 1080, max_width=960) == (960, 540)
