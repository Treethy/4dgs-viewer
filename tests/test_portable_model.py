from pathlib import Path

import torch

from viewer4d import Viewer4DGS, create_freetimegs_model


def make_model():
    n = 3
    return create_freetimegs_model(
        means=torch.zeros(n, 3),
        log_scales=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logits=torch.zeros(n, 1),
        sh0=torch.zeros(n, 1, 3),
        shN=torch.zeros(n, 15, 3),
        canonical_times=torch.tensor([[0.0], [0.5], [1.0]]),
        log_durations=torch.zeros(n, 1),
        velocities=torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1),
        num_frames=3,
        fps=12.0,
        source_metadata={"type": "unit-test"},
    )


def test_evaluate_frame():
    model = make_model()
    frame = model.evaluate_frame(1)
    assert frame.means.shape == (3, 3)
    assert torch.allclose(frame.means[:, 0], torch.tensor([0.5, 0.0, -0.5]))
    assert frame.scales.shape == (3, 3)
    assert frame.sh_coeffs.shape == (3, 16, 3)


def test_safe_roundtrip(tmp_path: Path):
    path = tmp_path / "model.v4d.pt"
    make_model().save(path)
    loaded = Viewer4DGS.load(path)
    frame = loaded.evaluate_frame(1)
    assert loaded.representation == "freetimegs.linear_temporal_gaussian.v1"
    assert torch.allclose(frame.means[:, 0], torch.tensor([0.5, 0.0, -0.5]))


def test_optional_save_during_import(tmp_path: Path):
    source_model = make_model()

    class FakeImporter:
        def convert(self, source, **kwargs):
            return source_model

    output = tmp_path / "converted.v4d.pt"
    loaded = Viewer4DGS.import_source(
        "source.pt",
        importer=FakeImporter(),
        save_converted_to=output,
    )
    assert output.exists()
    assert loaded.sequence.num_frames == 3
