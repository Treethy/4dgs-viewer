from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


_FORMAT_MAGIC = "anytimegs"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SequenceInfo:
    """Sequence metadata for an AnytimeGS model.

    AnytimeGS always uses a normalized evaluation time in [0, 1].
    Frame 0 maps to t=0 and frame num_frames-1 maps to t=1.
    """

    num_frames: int
    fps: float

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    def frame_to_time(self, frame: int) -> float:
        if not 0 <= frame < self.num_frames:
            raise IndexError(
                f"frame must be in [0, {self.num_frames - 1}], got {frame}"
            )
        if self.num_frames == 1:
            return 0.0
        return frame / (self.num_frames - 1)

    def time_to_frame(self, time: float) -> int:
        _validate_eval_time(time)
        if self.num_frames == 1:
            return 0
        return int(round(float(time) * (self.num_frames - 1)))

    def to_dict(self) -> dict[str, int | float]:
        return {
            "num_frames": self.num_frames,
            "fps": self.fps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceInfo":
        return cls(
            num_frames=int(value["num_frames"]),
            fps=float(value["fps"]),
        )


@dataclass(slots=True)
class GaussianFrame:
    """Render-ready static Gaussians at one normalized time."""

    means: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    sh: Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self.means.device

    def validate(self) -> None:
        _require_shape("means", self.means, 2, last_dim=3)
        n = self.means.shape[0]
        _require_exact_shape("scales", self.scales, (n, 3))
        _require_exact_shape("quats", self.quats, (n, 4))

        if self.opacities.ndim == 2 and self.opacities.shape == (n, 1):
            self.opacities = self.opacities[:, 0]
        _require_exact_shape("opacities", self.opacities, (n,))

        if self.sh.ndim != 3 or self.sh.shape[0] != n or self.sh.shape[-1] != 3:
            raise ValueError(f"sh must have shape [N, K, 3], got {tuple(self.sh.shape)}")

        _require_same_device(
            self.means,
            self.scales,
            self.quats,
            self.opacities,
            self.sh,
        )
        _require_floating(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh=self.sh,
        )
        _require_finite(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh=self.sh,
        )

        if torch.any(self.scales <= 0):
            raise ValueError("scales must be strictly positive")
        if torch.any((self.opacities < 0) | (self.opacities > 1)):
            raise ValueError("opacities must be in [0, 1]")


class AnytimeGS:
    """Fixed-schema portable 4D Gaussian representation.

    Stored parameters are source-independent, activated values:

    - means:          [N, 3], Gaussian position at ``time_center``.
    - scales:         [N, 3], positive Gaussian scales (not log-scales).
    - quats:          [N, 4], rotation quaternions.
    - opacities:      [N], base opacity in [0, 1] (not logits).
    - sh:             [N, K, 3], spherical-harmonic coefficients.
    - time_center:    [N], center time in normalized time coordinates.
    - duration:       [N], positive temporal standard deviation in normalized
                              time coordinates. It may be greater than 1.
    - velocity:       [N, 3], displacement per one normalized time unit. Its
                              magnitude is not bounded and may be greater than 1.
    - temporal_gate:  [N], temporal-opacity floor in [0, 1]. A value close to
                              1 makes a Gaussian effectively persistent.

    Evaluation time is always constrained to [0, 1]. ``time_center`` itself is
    not clamped: source models may legitimately optimize centers outside the
    observed interval, as long as they are expressed in the normalized time
    coordinate system.
    """

    def __init__(
        self,
        *,
        sequence: SequenceInfo,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
        opacities: Tensor,
        sh: Tensor,
        time_center: Tensor,
        duration: Tensor,
        velocity: Tensor,
        temporal_gate: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.sequence = sequence
        self.means = means
        self.scales = scales
        self.quats = quats
        self.opacities = _flatten_scalar_parameter("opacities", opacities)
        self.sh = sh
        self.time_center = _flatten_scalar_parameter("time_center", time_center)
        self.duration = _flatten_scalar_parameter("duration", duration)
        self.velocity = velocity

        if temporal_gate is None:
            temporal_gate = torch.zeros_like(self.opacities)
        self.temporal_gate = _flatten_scalar_parameter(
            "temporal_gate", temporal_gate
        )
        self.metadata = _validate_metadata(dict(metadata or {}))

        self.validate()

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self.means.device

    def validate(self) -> None:
        _require_shape("means", self.means, 2, last_dim=3)
        n = self.means.shape[0]

        _require_exact_shape("scales", self.scales, (n, 3))
        _require_exact_shape("quats", self.quats, (n, 4))
        _require_exact_shape("opacities", self.opacities, (n,))
        _require_exact_shape("time_center", self.time_center, (n,))
        _require_exact_shape("duration", self.duration, (n,))
        _require_exact_shape("velocity", self.velocity, (n, 3))
        _require_exact_shape("temporal_gate", self.temporal_gate, (n,))

        if self.sh.ndim != 3 or self.sh.shape[0] != n or self.sh.shape[-1] != 3:
            raise ValueError(f"sh must have shape [N, K, 3], got {tuple(self.sh.shape)}")

        tensors = {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh": self.sh,
            "time_center": self.time_center,
            "duration": self.duration,
            "velocity": self.velocity,
            "temporal_gate": self.temporal_gate,
        }

        _require_same_device(*tensors.values())
        _require_floating(**tensors)
        _require_finite(**tensors)

        if torch.any(self.scales <= 0):
            raise ValueError("scales must be strictly positive")
        if torch.any((self.opacities < 0) | (self.opacities > 1)):
            raise ValueError("opacities must be in [0, 1]")
        if torch.any(self.duration <= 0):
            raise ValueError("duration must be strictly positive")
        if torch.any((self.temporal_gate < 0) | (self.temporal_gate > 1)):
            raise ValueError("temporal_gate must be in [0, 1]")

        quat_norm = torch.linalg.vector_norm(self.quats, dim=-1)
        if torch.any(quat_norm <= torch.finfo(self.quats.dtype).eps):
            raise ValueError("quats must have non-zero norm")

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> "AnytimeGS":
        """Move all Gaussian tensors in place and return ``self``."""

        for name in self._tensor_names():
            value = getattr(self, name)
            setattr(
                self,
                name,
                value.to(device=device, non_blocking=non_blocking),
            )
        return self

    @torch.inference_mode()
    def at_time(self, time: float) -> GaussianFrame:
        """Evaluate the 4D Gaussians at normalized time ``time`` in [0, 1]."""

        _validate_eval_time(time)
        t = torch.as_tensor(float(time), dtype=self.means.dtype, device=self.device)

        dt = t - self.time_center
        means = self.means + dt[:, None] * self.velocity

        temporal_weight = self.temporal_gate + (1.0 - self.temporal_gate) * torch.exp(
            -0.5 * (dt / self.duration).square()
        )
        opacities = self.opacities * temporal_weight

        quats = torch.nn.functional.normalize(self.quats, dim=-1)

        return GaussianFrame(
            means=means,
            scales=self.scales,
            quats=quats,
            opacities=opacities,
            sh=self.sh,
        )

    @torch.inference_mode()
    def at_frame(self, frame: int) -> GaussianFrame:
        return self.at_time(self.sequence.frame_to_time(frame))

    def to_payload(self, *, cpu: bool = True) -> dict[str, Any]:
        """Return a fixed-schema, weights-only-safe serialization payload."""

        def prepare(value: Tensor) -> Tensor:
            value = value.detach()
            if cpu:
                value = value.cpu()
            return value.contiguous()

        gaussians = {
            name: prepare(getattr(self, name)) for name in self._tensor_names()
        }

        return {
            "format": _FORMAT_MAGIC,
            "schema_version": _SCHEMA_VERSION,
            "sequence": self.sequence.to_dict(),
            "gaussians": gaussians,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically save this AnytimeGS model as CPU tensors."""

        destination = Path(path).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {destination}; "
                "pass overwrite=True to replace it"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)

        try:
            torch.save(self.to_payload(cpu=True), temporary_path)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        return destination

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> "AnytimeGS":
        _validate_payload(payload)

        gaussians = payload["gaussians"]
        model = cls(
            sequence=SequenceInfo.from_dict(payload["sequence"]),
            means=gaussians["means"],
            scales=gaussians["scales"],
            quats=gaussians["quats"],
            opacities=gaussians["opacities"],
            sh=gaussians["sh"],
            time_center=gaussians["time_center"],
            duration=gaussians["duration"],
            velocity=gaussians["velocity"],
            temporal_gate=gaussians["temporal_gate"],
            metadata=payload.get("metadata", {}),
        )
        return model.to(device)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        mmap: bool = False,
    ) -> "AnytimeGS":
        """Load an AnytimeGS file without any source-project dependency."""

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
            mmap=mmap,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("AnytimeGS file root must be a mapping")
        return cls.from_payload(payload, device=device)

    @staticmethod
    def _tensor_names() -> tuple[str, ...]:
        return (
            "means",
            "scales",
            "quats",
            "opacities",
            "sh",
            "time_center",
            "duration",
            "velocity",
            "temporal_gate",
        )


def _validate_eval_time(time: float) -> None:
    value = float(time)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"time must be in [0, 1], got {time}")


def _flatten_scalar_parameter(name: str, value: Tensor) -> Tensor:
    if value.ndim == 2 and value.shape[1] == 1:
        return value[:, 0]
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N] or [N, 1], got {tuple(value.shape)}")
    return value


def _require_shape(name: str, value: Tensor, ndim: int, *, last_dim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim or value.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have shape [N, {last_dim}], got {tuple(value.shape)}"
        )


def _require_exact_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def _require_same_device(*values: Tensor) -> None:
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError(f"all Gaussian tensors must share one device, got {devices}")


def _require_floating(**values: Tensor) -> None:
    for name, value in values.items():
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype, got {value.dtype}")


def _require_finite(**values: Tensor) -> None:
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")


def _validate_metadata(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _validate_metadata(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _validate_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only JSON-like values, got {type(value).__name__}"
    )


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != _FORMAT_MAGIC:
        raise ValueError(
            f"not an AnytimeGS file: expected format={_FORMAT_MAGIC!r}, "
            f"got {payload.get('format')!r}"
        )
    if int(payload.get("schema_version", -1)) != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported AnytimeGS schema version: {payload.get('schema_version')!r}"
        )

    sequence = payload.get("sequence")
    if not isinstance(sequence, Mapping):
        raise ValueError("payload['sequence'] must be a mapping")

    gaussians = payload.get("gaussians")
    if not isinstance(gaussians, Mapping):
        raise ValueError("payload['gaussians'] must be a mapping")

    required = set(AnytimeGS._tensor_names())
    actual = set(gaussians.keys())
    missing = required - actual
    extra = actual - required
    if missing:
        raise ValueError(f"missing Gaussian fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected Gaussian fields: {sorted(extra)}")

    for name in required:
        if not isinstance(gaussians[name], Tensor):
            raise TypeError(f"gaussians[{name!r}] must be a torch.Tensor")

    _validate_metadata(payload.get("metadata", {}))