from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from viewer4d import FreeTimeGSImporter, Viewer4DGS


def _extracted_payload(n: int = 4):
    return {
        "format": "viewer4d.freetimegs_extracted",
        "schema_version": 1,
        "tensors": {
            "means": torch.zeros(n, 3),
            "log_scales": torch.zeros(n, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            "opacity_logits": torch.zeros(n, 1),
            "sh0": torch.zeros(n, 1, 3),
            "shN": torch.zeros(n, 15, 3),
            "canonical_times": torch.zeros(n, 1),
            "log_durations": torch.zeros(n, 1),
            "velocities": torch.ones(n, 3),
        },
        "extras": {"marginal_gates": torch.ones(n, 1)},
        "source_metadata": {"source_class": "fake.Gaussians"},
    }


def test_importer_calls_worker_and_optionally_saves(tmp_path: Path, monkeypatch):
    source_project = tmp_path / "ftgs"
    source_project.mkdir()
    source = tmp_path / "gaussians.pt"
    source.write_bytes(b"placeholder")

    def fake_run(command, **kwargs):
        output_index = command.index("--output") + 1
        torch.save(_extracted_payload(), Path(command[output_index]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    importer = FreeTimeGSImporter(
        source_project=source_project,
        trust_source=True,
    )
    output = tmp_path / "model.v4d.pt"
    model = Viewer4DGS.import_source(
        source,
        importer=importer,
        save_converted_to=output,
        num_frames=10,
        fps=12.0,
    )

    assert output.is_file()
    assert model.sequence.num_frames == 10
    assert model.tensor("base.means").shape == (4, 3)
    assert model.tensor("extras.marginal_gates").shape == (4, 1)

    reloaded = Viewer4DGS.load(output)
    assert reloaded.tensor("motion.velocities").shape == (4, 3)


def test_importer_requires_explicit_trust(tmp_path: Path):
    source_project = tmp_path / "ftgs"
    source_project.mkdir()
    source = tmp_path / "gaussians.pt"
    source.write_bytes(b"placeholder")

    importer = FreeTimeGSImporter(
        source_project=source_project,
        trust_source=False,
    )
    with pytest.raises(PermissionError):
        importer.convert(source, num_frames=10, fps=12.0)


def test_worker_extracts_custom_module_object(tmp_path: Path):
    module_path = tmp_path / "fake_ftgs.py"
    module_path.write_text(
        """
import torch

class Gaussians(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n = 2
        self.means = torch.nn.Parameter(torch.zeros(n, 3))
        self.scales = torch.nn.Parameter(torch.zeros(n, 3))
        self.quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1))
        self.opacities = torch.nn.Parameter(torch.zeros(n, 1))
        self.sh_0 = torch.nn.Parameter(torch.zeros(n, 1, 3))
        self.sh_n = torch.nn.Parameter(torch.zeros(n, 15, 3))
        self.times = torch.nn.Parameter(torch.zeros(n, 1))
        self.durations = torch.nn.Parameter(torch.zeros(n, 1))
        self.velocity_model = torch.nn.Parameter(torch.ones(n, 3))
        self.marginal_gates = torch.nn.Parameter(torch.ones(n, 1), requires_grad=False)
        self.sh_degree = 3
""".strip()
    )

    import importlib.util
    import os
    import sys

    spec = importlib.util.spec_from_file_location("fake_ftgs", module_path)
    assert spec is not None and spec.loader is not None
    fake_ftgs = importlib.util.module_from_spec(spec)
    sys.modules["fake_ftgs"] = fake_ftgs
    spec.loader.exec_module(fake_ftgs)

    source = tmp_path / "source.pt"
    torch.save(fake_ftgs.Gaussians(), source)
    output = tmp_path / "extracted.pt"
    worker = (
        Path(__file__).parents[1]
        / "src/viewer4d/importers/workers/freetimegs_export.py"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    payload = torch.load(output, weights_only=True)
    assert payload["tensors"]["means"].shape == (2, 3)
    assert payload["extras"]["marginal_gates"].shape == (2, 1)
