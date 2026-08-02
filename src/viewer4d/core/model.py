from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypeAlias

import torch
from torch import Tensor

TensorTree: TypeAlias = dict[str, Any]
MetadataValue: TypeAlias = Any
Evaluator: TypeAlias = Callable[["Viewer4DGS", float], "GaussianFrame"]

_FORMAT_MAGIC = "viewer4d.portable_model"
_SCHEMA_VERSION = 1
_EVALUATORS: dict[str, Evaluator] = {}


@dataclass(frozen=True, slots=True)
class SequenceInfo:
    """Temporal information shared by every portable 4DGS model."""

    num_frames: int
    fps: float
    time_min: float = 0.0
    time_max: float = 1.0

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.time_max < self.time_min:
            raise ValueError("time_max must be greater than or equal to time_min")

    def frame_to_time(self, frame: int) -> float:
        if not 0 <= frame < self.num_frames:
            raise IndexError(
                f"frame must be in [0, {self.num_frames - 1}], got {frame}"
            )
        if self.num_frames == 1:
            return self.time_min
        ratio = frame / (self.num_frames - 1)
        return self.time_min + ratio * (self.time_max - self.time_min)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "num_frames": self.num_frames,
            "fps": self.fps,
            "time_min": self.time_min,
            "time_max": self.time_max,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceInfo":
        return cls(
            num_frames=int(value["num_frames"]),
            fps=float(value["fps"]),
            time_min=float(value.get("time_min", 0.0)),
            time_max=float(value.get("time_max", 1.0)),
        )


@dataclass(slots=True)
class GaussianFrame:
    """Render-ready static Gaussians evaluated at one time instant."""

    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    sh_coeffs: Tensor

    def validate(self) -> None:
        if self.means.ndim != 2 or self.means.shape[1] != 3:
            raise ValueError(f"means must have shape [N, 3], got {self.means.shape}")
        n = self.means.shape[0]
        expected = {
            "quats": (n, 4),
            "scales": (n, 3),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tensor.shape}")

        if self.opacities.ndim == 2 and self.opacities.shape[1] == 1:
            self.opacities = self.opacities[:, 0]
        if tuple(self.opacities.shape) != (n,):
            raise ValueError(
                f"opacities must have shape [N] or [N, 1], got {self.opacities.shape}"
            )
        if self.sh_coeffs.ndim != 3 or self.sh_coeffs.shape[0] != n:
            raise ValueError(
                "sh_coeffs must have shape [N, K, 3], "
                f"got {self.sh_coeffs.shape}"
            )
        if self.sh_coeffs.shape[2] != 3:
            raise ValueError(
                f"sh_coeffs last dimension must be 3, got {self.sh_coeffs.shape}"
            )

        devices = {
            self.means.device,
            self.quats.device,
            self.scales.device,
            self.opacities.device,
            self.sh_coeffs.device,
        }
        if len(devices) != 1:
            raise ValueError(f"all frame tensors must share one device, got {devices}")


class ModelImporter(Protocol):
    """Interface implemented by source-format importers.

    An importer may execute in another Python environment. Its final result must
    be either a Viewer4DGS instance or a portable payload containing only tensors
    and primitive metadata.
    """

    def convert(self, source: str | Path, **kwargs: Any) -> "Viewer4DGS | Mapping[str, Any]":
        ...


def register_evaluator(name: str) -> Callable[[Evaluator], Evaluator]:
    """Register the runtime evaluator for a portable representation."""

    if not name or not isinstance(name, str):
        raise ValueError("representation name must be a non-empty string")

    def decorator(function: Evaluator) -> Evaluator:
        if name in _EVALUATORS:
            raise ValueError(f"evaluator already registered: {name}")
        _EVALUATORS[name] = function
        return function

    return decorator


class Viewer4DGS:
    """Portable 4D Gaussian model used by 4Dviewer.

    The class deliberately stores only:
      * a representation identifier;
      * sequence metadata;
      * a nested tree of tensors;
      * JSON-like metadata.

    Source-project classes are never serialized. This lets converted files be
    loaded in the viewer environment with ``torch.load(weights_only=True)``.
    """

    def __init__(
        self,
        *,
        representation: str,
        sequence: SequenceInfo,
        tensors: Mapping[str, Tensor | Mapping[str, Any]],
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> None:
        if not representation:
            raise ValueError("representation must be a non-empty string")
        self.representation = representation
        self.sequence = sequence
        self.tensors = _copy_tensor_tree(tensors)
        self.metadata: dict[str, MetadataValue] = _validate_metadata(metadata or {})
        if not self.tensors:
            raise ValueError("tensors must not be empty")

    @property
    def device(self) -> torch.device:
        first = next(_iter_tensors(self.tensors), None)
        if first is None:
            raise RuntimeError("model contains no tensors")
        return first.device

    def tensor(self, path: str) -> Tensor:
        """Read a tensor using a dotted path, e.g. ``base.means``."""

        current: Any = self.tensors
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"tensor path not found: {path}")
            current = current[part]
        if not isinstance(current, Tensor):
            raise KeyError(f"path points to a tensor group, not a tensor: {path}")
        return current

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> "Viewer4DGS":
        """Move all stored tensors in place and return ``self``."""

        self.tensors = _map_tensor_tree(
            self.tensors,
            lambda value: value.to(device=device, non_blocking=non_blocking),
        )
        return self

    @torch.inference_mode()
    def evaluate_time(self, time: float) -> GaussianFrame:
        if not self.sequence.time_min <= time <= self.sequence.time_max:
            raise ValueError(
                f"time must be in [{self.sequence.time_min}, "
                f"{self.sequence.time_max}], got {time}"
            )
        try:
            evaluator = _EVALUATORS[self.representation]
        except KeyError as error:
            available = ", ".join(sorted(_EVALUATORS)) or "<none>"
            raise RuntimeError(
                f"No evaluator registered for representation "
                f"'{self.representation}'. Available: {available}"
            ) from error
        frame = evaluator(self, float(time))
        frame.validate()
        return frame

    @torch.inference_mode()
    def evaluate_frame(self, frame: int) -> GaussianFrame:
        return self.evaluate_time(self.sequence.frame_to_time(frame))

    def to_payload(self, *, cpu: bool = True) -> dict[str, Any]:
        """Create a pickle-independent, weights-only-safe payload."""

        def prepare(tensor: Tensor) -> Tensor:
            value = tensor.detach()
            if cpu:
                value = value.cpu()
            return value.contiguous()

        return {
            "format": _FORMAT_MAGIC,
            "schema_version": _SCHEMA_VERSION,
            "representation": self.representation,
            "sequence": self.sequence.to_dict(),
            "tensors": _map_tensor_tree(self.tensors, prepare),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically save a portable model as CPU tensors."""

        destination = Path(path).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {destination}; pass overwrite=True to replace it"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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
    ) -> "Viewer4DGS":
        _validate_payload(payload)
        model = cls(
            representation=str(payload["representation"]),
            sequence=SequenceInfo.from_dict(payload["sequence"]),
            tensors=payload["tensors"],
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
    ) -> "Viewer4DGS":
        """Safely load a converted model without source-project dependencies."""

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
            raise ValueError("portable model root must be a mapping")
        return cls.from_payload(payload, device=device)

    @classmethod
    def import_source(
        cls,
        source: str | Path,
        *,
        importer: ModelImporter,
        save_converted_to: str | Path | None = None,
        overwrite: bool = False,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "Viewer4DGS":
        """Convert a source model and optionally persist the portable result.

        The importer owns source-specific loading. It can run the original model
        in another uv environment and return a plain portable payload.
        """

        converted = importer.convert(source, **kwargs)
        if isinstance(converted, cls):
            model = converted
        elif isinstance(converted, Mapping):
            model = cls.from_payload(converted, device="cpu")
        else:
            raise TypeError(
                "importer.convert() must return Viewer4DGS or a portable payload"
            )

        if save_converted_to is not None:
            model.save(save_converted_to, overwrite=overwrite)
        return model.to(device)


def _copy_tensor_tree(value: Mapping[str, Any]) -> TensorTree:
    result: TensorTree = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("tensor tree keys must be non-empty strings")
        if isinstance(item, Tensor):
            result[key] = item
        elif isinstance(item, Mapping):
            result[key] = _copy_tensor_tree(item)
        else:
            raise TypeError(
                f"tensor tree value at '{key}' must be a Tensor or mapping, "
                f"got {type(item).__name__}"
            )
    return result


def _map_tensor_tree(
    tree: TensorTree,
    function: Callable[[Tensor], Tensor],
) -> TensorTree:
    result: TensorTree = {}
    for key, item in tree.items():
        if isinstance(item, Tensor):
            result[key] = function(item)
        else:
            result[key] = _map_tensor_tree(item, function)
    return result


def _iter_tensors(tree: TensorTree):
    for item in tree.values():
        if isinstance(item, Tensor):
            yield item
        else:
            yield from _iter_tensors(item)


def _validate_metadata(value: Mapping[str, Any]) -> dict[str, MetadataValue]:
    def convert(item: Any, path: str) -> MetadataValue:
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (list, tuple)):
            return [convert(child, f"{path}[]") for child in item]
        if isinstance(item, Mapping):
            output: dict[str, MetadataValue] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"metadata key at {path} must be a string")
                output[key] = convert(child, f"{path}.{key}")
            return output
        raise TypeError(
            f"metadata at {path} must be JSON-like, got {type(item).__name__}"
        )

    return {str(key): convert(item, f"metadata.{key}") for key, item in value.items()}


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != _FORMAT_MAGIC:
        raise ValueError("not a viewer4d portable model")
    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version}; expected {_SCHEMA_VERSION}"
        )
    required = ("representation", "sequence", "tensors")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"portable model is missing fields: {', '.join(missing)}")
    if not isinstance(payload["sequence"], Mapping):
        raise TypeError("sequence must be a mapping")
    if not isinstance(payload["tensors"], Mapping):
        raise TypeError("tensors must be a mapping")
