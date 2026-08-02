from pathlib import Path

import pytest
import torch

from viewer4d import Viewer4DGS, create_freetimegs_model
from viewer4d.visualization.loading import ModelLoadError, load_viewer_model


def _make_model():
    n = 2
    return create_freetimegs_model(
        means=torch.zeros(n, 3),
        log_scales=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logits=torch.zeros(n, 1),
        sh0=torch.zeros(n, 1, 3),
        shN=torch.zeros(n, 15, 3),
        canonical_times=torch.zeros(n, 1),
        log_durations=torch.zeros(n, 1),
        velocities=torch.zeros(n, 3),
        num_frames=5,
        fps=12.0,
    )


def test_load_portable_auto(tmp_path: Path):
    path = tmp_path / "scene.v4d.pt"
    _make_model().save(path)
    loaded = load_viewer_model(path, input_format="auto")
    assert isinstance(loaded, Viewer4DGS)
    assert loaded.sequence.num_frames == 5


def test_auto_reports_original_format_options(tmp_path: Path):
    path = tmp_path / "unknown.pt"
    path.write_bytes(b"not a portable checkpoint")
    with pytest.raises(ModelLoadError, match="source_project"):
        load_viewer_model(path, input_format="auto")
