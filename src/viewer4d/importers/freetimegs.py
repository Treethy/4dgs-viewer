from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from viewer4d.core.model import Viewer4DGS
from viewer4d.importers.base import BaseImporter
from viewer4d.representations.freetimegs import create_freetimegs_model

_EXTRACT_FORMAT = "viewer4d.freetimegs_extracted"
_EXTRACT_VERSION = 1
_REQUIRED_TENSORS = (
    "means",
    "log_scales",
    "quats",
    "opacity_logits",
    "sh0",
    "shN",
    "canonical_times",
    "log_durations",
    "velocities",
)


class FreeTimeGSImporter(BaseImporter):
    """Import a FreeTimeGS/FreeTimeGS++ checkpoint through its own uv project.

    The source checkpoint may contain custom Python objects and CUDA-extension
    imports. Therefore it is decoded by a small worker launched inside the
    source project's uv environment. The worker returns only CPU tensors and
    primitive metadata, which this viewer environment can safely load.
    """

    def __init__(
        self,
        *,
        source_project: str | Path,
        trust_source: bool = False,
        uv_executable: str = "uv",
        no_sync: bool = True,
    ) -> None:
        self.source_project = Path(source_project).expanduser().resolve()
        self.trust_source = trust_source
        self.uv_executable = uv_executable
        self.no_sync = no_sync

        if not self.source_project.is_dir():
            raise NotADirectoryError(self.source_project)

    def convert(
        self,
        source: str | Path,
        **kwargs: Any,
    ) -> Viewer4DGS:
        """Convert one original checkpoint into the existing portable model.

        Required keyword arguments:
            num_frames: Number of frames represented by the normalized timeline.
            fps: Playback frame rate used by the viewer.

        Optional keyword arguments:
            time_min/time_max: Stored normalized time range. The current
                FreeTimeGS representation uses [0, 1], so other values are
                rejected for now rather than silently producing wrong motion.
        """

        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if not self.trust_source:
            raise PermissionError(
                "FreeTimeGS checkpoints require torch.load(weights_only=False). "
                "Construct FreeTimeGSImporter(..., trust_source=True) only for "
                "a checkpoint you trust."
            )

        try:
            num_frames = int(kwargs.pop("num_frames"))
            fps = float(kwargs.pop("fps"))
        except KeyError as error:
            raise TypeError(f"Missing required importer option: {error.args[0]}") from error

        time_min = float(kwargs.pop("time_min", 0.0))
        time_max = float(kwargs.pop("time_max", 1.0))
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected FreeTimeGS importer options: {names}")
        if (time_min, time_max) != (0.0, 1.0):
            raise ValueError(
                "The current FreeTimeGS portable representation expects the "
                "normalized time range [0, 1]."
            )

        extracted = self._run_worker(source_path)
        tensors, extras, source_metadata = _validate_extracted_payload(extracted)

        return create_freetimegs_model(
            means=tensors["means"],
            log_scales=tensors["log_scales"],
            quats=tensors["quats"],
            opacity_logits=tensors["opacity_logits"],
            sh0=tensors["sh0"],
            shN=tensors["shN"],
            canonical_times=tensors["canonical_times"],
            log_durations=tensors["log_durations"],
            velocities=tensors["velocities"],
            extras=extras or None,
            num_frames=num_frames,
            fps=fps,
            source_metadata={
                **source_metadata,
                "source_project": str(self.source_project),
                "importer": "viewer4d.FreeTimeGSImporter",
            },
        )

    def _run_worker(self, source: Path) -> Mapping[str, Any]:
        worker = Path(__file__).resolve().parent / "workers" / "freetimegs_export.py"
        if not worker.is_file():
            raise FileNotFoundError(f"FreeTimeGS importer worker not found: {worker}")

        fd, temporary_name = tempfile.mkstemp(
            prefix="viewer4d-ftgs-", suffix=".pt"
        )
        os.close(fd)
        temporary_path = Path(temporary_name)

        command = [
            self.uv_executable,
            "run",
            "--project",
            str(self.source_project),
        ]
        if self.no_sync:
            command.append("--no-sync")
        command.extend(
            [
                "python",
                str(worker),
                "--input",
                str(source),
                "--output",
                str(temporary_path),
            ]
        )

        try:
            completed = subprocess.run(
                command,
                cwd=self.source_project,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                details = "\n".join(
                    part.strip()
                    for part in (completed.stdout, completed.stderr)
                    if part.strip()
                )
                raise RuntimeError(
                    "FreeTimeGS source-environment worker failed.\n"
                    f"Command: {' '.join(command)}\n"
                    f"Output:\n{details or '<no output>'}"
                )

            payload = torch.load(
                temporary_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(payload, Mapping):
                raise TypeError("FreeTimeGS worker output root must be a mapping")
            return payload
        finally:
            temporary_path.unlink(missing_ok=True)


def _validate_extracted_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    if payload.get("format") != _EXTRACT_FORMAT:
        raise ValueError("FreeTimeGS worker returned an unknown payload format")
    if payload.get("schema_version") != _EXTRACT_VERSION:
        raise ValueError(
            "Unsupported FreeTimeGS extraction schema version: "
            f"{payload.get('schema_version')}"
        )

    raw_tensors = payload.get("tensors")
    if not isinstance(raw_tensors, Mapping):
        raise TypeError("FreeTimeGS worker payload.tensors must be a mapping")

    tensors: dict[str, Tensor] = {}
    for name in _REQUIRED_TENSORS:
        value = raw_tensors.get(name)
        if not isinstance(value, Tensor):
            raise TypeError(f"FreeTimeGS worker tensor '{name}' is missing or invalid")
        tensors[name] = value.detach().cpu().contiguous()

    raw_extras = payload.get("extras", {})
    if not isinstance(raw_extras, Mapping):
        raise TypeError("FreeTimeGS worker payload.extras must be a mapping")
    extras: dict[str, Tensor] = {}
    for name, value in raw_extras.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("FreeTimeGS extras must map string names to tensors")
        extras[name] = value.detach().cpu().contiguous()

    raw_metadata = payload.get("source_metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("FreeTimeGS source_metadata must be a mapping")
    source_metadata = {str(key): value for key, value in raw_metadata.items()}

    _validate_shapes(tensors)
    return tensors, extras, source_metadata


def _validate_shapes(tensors: Mapping[str, Tensor]) -> None:
    means = tensors["means"]
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must have shape [N, 3], got {tuple(means.shape)}")
    n = means.shape[0]

    expected_shapes = {
        "log_scales": (n, 3),
        "quats": (n, 4),
        "opacity_logits": (n, 1),
        "canonical_times": (n, 1),
        "log_durations": (n, 1),
        "velocities": (n, 3),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}")

    for name in ("sh0", "shN"):
        tensor = tensors[name]
        if tensor.ndim != 3 or tensor.shape[0] != n or tensor.shape[2] != 3:
            raise ValueError(
                f"{name} must have shape [N, K, 3], got {tuple(tensor.shape)}"
            )
