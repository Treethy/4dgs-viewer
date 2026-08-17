from __future__ import annotations

import ast
import pickle
import re
import tomllib
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from viewer4d.conversion.transfer import freetimegs_to_anytimegs
from viewer4d.core.camera import PinholeCamera
from viewer4d.core.model import AnytimeGS


class _FreeTimeGSGaussians(nn.Module):
    """Compatibility shell for ftgspp.models.gaussians.Gaussians."""


class _FreeTimeGSUnpickler(pickle.Unpickler):
    """Deserialize the explicit-velocity FTGS++ checkpoint without importing FTGS++."""

    _SOURCE_CLASS = ("ftgspp.models.gaussians", "Gaussians")
    _ALLOWED_GLOBALS = {
        ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch", "FloatStorage"): torch.FloatStorage,
        ("collections", "OrderedDict"): OrderedDict,
        ("__builtin__", "set"): set,
        ("builtins", "set"): set,
    }

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key == self._SOURCE_CLASS:
            return _FreeTimeGSGaussians

        if key in self._ALLOWED_GLOBALS:
            return self._ALLOWED_GLOBALS[key]

        raise pickle.UnpicklingError(
            f"unsupported global in FreeTimeGS++ checkpoint: {module}.{name}"
        )


_FREETIMEGS_PICKLE = types.SimpleNamespace(
    __name__="viewer4d_freetimegs_compat_pickle",
    Unpickler=_FreeTimeGSUnpickler,
    Pickler=pickle.Pickler,
    load=pickle.load,
    loads=pickle.loads,
    dump=pickle.dump,
    dumps=pickle.dumps,
    HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL,
)


class FreeTimeGSImporter:
    """FreeTimeGS++ file adapter.

    This class owns source-format concerns only:
      - checkpoint compatibility deserialization
      - TOML loading, including FTGS++ `extends`
      - conversion to AnytimeGS
    """

    def convert(
        self,
        checkpoint: str | Path,
        *,
        config: str | Path,
        device: str | torch.device = "cpu",
    ) -> AnytimeGS:
        checkpoint_data = self.load_checkpoint(checkpoint)
        config_data = load_freetimegs_config(config)

        return freetimegs_to_anytimegs(
            checkpoint_data,
            config_data,
            device=device,
        )

    @staticmethod
    def load_checkpoint(path: str | Path) -> _FreeTimeGSGaussians:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        source = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_FREETIMEGS_PICKLE,
        )
        _validate_checkpoint(source)
        return source


def load_freetimegs_config(path: str | Path) -> dict[str, Any]:
    """Load an FTGS++ TOML config and recursively resolve `extends`.

    FTGS++ resolves `extends` relative to the current TOML file, so we preserve
    that behavior instead of relying on plain tomllib.load().
    """

    path = Path(path).expanduser().resolve()
    return _load_config_recursive(path, seen=set())


def load_freetimegs_eval_camera(
    *,
    config: str | Path | Mapping[str, Any],
    intrinsics: str | Path,
    extrinsics: str | Path,
) -> PinholeCamera:
    """Build the initial viewer camera from the first FTGS++ eval camera.

    The camera index comes from `data.eval_cameras`. Camera calibration comes
    from the passed OpenCV intri.yml / extri.yml.

    Intrinsics reproduce the same undistortion/cropping/scale calculation used
    by FreeTimeGS++ when preparing MultiViewData.
    """

    config_data = (
        load_freetimegs_config(config)
        if isinstance(config, (str, Path))
        else dict(config)
    )

    camera_index = _first_eval_camera(config_data)

    intri = _OpenCVIntrinsics.read(intrinsics)
    extri = _OpenCVExtrinsics.read(extrinsics)

    if set(intri.names) != set(extri.names):
        raise ValueError("intri.yml and extri.yml camera names do not match")

    camera_name = _camera_name_from_index(camera_index, intri.names)

    scale = float(config_data["data"].get("scale", 1.0))
    if scale <= 0:
        raise ValueError(f"data.scale must be positive, got {scale}")

    processed_K, width, height = _processed_intrinsics(
        intri,
        camera_name=camera_name,
        scale=scale,
    )

    R = extri.rots[camera_name]
    T = extri.ts[camera_name]

    # FTGS++ passes Rot_<name> / T_<name> to pycolmap.Rigid3d as cam_from_world.
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    c2w = np.linalg.inv(w2c)

    return PinholeCamera(
        c2w=c2w,
        K=processed_K,
        width=width,
        height=height,
        name=camera_name,
    )


def _load_config_recursive(
    path: Path,
    *,
    seen: set[Path],
) -> dict[str, Any]:
    path = path.expanduser().resolve()

    if path in seen:
        raise ValueError(f"cyclic FreeTimeGS++ config extends detected at {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    seen = set(seen)
    seen.add(path)

    with path.open("rb") as file:
        current = tomllib.load(file)

    extends = current.pop("extends", None)
    if extends is None:
        return current

    base_path = (path.parent / str(extends)).resolve()
    base = _load_config_recursive(base_path, seen=seen)
    return _deep_update(base, current)


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)

    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(dict(result[key]), value)
        else:
            result[key] = value

    return result


def _first_eval_camera(config: Mapping[str, Any]) -> int:
    try:
        selection = config["data"]["eval_cameras"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain data.eval_cameras") from error

    if isinstance(selection, list):
        if not selection:
            raise ValueError("data.eval_cameras must not be empty")
        camera = int(selection[0])
    elif isinstance(selection, Mapping):
        try:
            start = int(selection["start"])
            stop = int(selection["stop"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "interval data.eval_cameras requires {start, stop}"
            ) from error

        if stop <= start:
            raise ValueError(
                f"invalid data.eval_cameras interval: [{start}, {stop})"
            )
        camera = start
    else:
        raise ValueError(
            "data.eval_cameras must be a list[int] or {start, stop} interval"
        )

    if camera < 0:
        raise ValueError(f"eval camera index must be non-negative, got {camera}")

    return camera


def _camera_name_from_index(index: int, names: list[str]) -> str:
    # SelfCap uses names such as 0000, 0001, ... and eval_cameras uses the
    # numeric camera ID. Prefer that exact correspondence.
    numeric_matches = [name for name in names if name.isdigit() and int(name) == index]
    if len(numeric_matches) == 1:
        return numeric_matches[0]
    if len(numeric_matches) > 1:
        raise ValueError(f"ambiguous calibration names for camera {index}")

    # Fallback for calibration files whose names are not zero-padded IDs.
    ordered = sorted(names, key=_camera_sort_key)
    if 0 <= index < len(ordered):
        return ordered[index]

    raise IndexError(
        f"eval camera {index} cannot be mapped to calibration names {names}"
    )


def _camera_sort_key(name: str) -> tuple[int, int | str]:
    if name.isdigit():
        return (0, int(name))
    return (1, name)



@dataclass(frozen=True, slots=True)
class _OpenCVIntrinsics:
    names: list[str]
    ks: dict[str, np.ndarray]
    shapes: dict[str, tuple[int, int]]  # H, W

    @classmethod
    def read(cls, path: str | Path) -> "_OpenCVIntrinsics":
        document = _read_opencv_yaml(path)

        ks: dict[str, np.ndarray] = {}
        shapes: dict[str, tuple[int, int]] = {}

        for name in document.names:
            K = document.matrices.get(f"K_{name}")
            H = document.scalars.get(f"H_{name}")
            W = document.scalars.get(f"W_{name}")

            if K is None or H is None or W is None:
                raise ValueError(
                    f"missing K/H/W calibration for camera {name}"
                )
            if K.shape != (3, 3):
                raise ValueError(
                    f"K_{name} must be 3x3, got {K.shape}"
                )

            ks[name] = K.astype(np.float64, copy=True)
            shapes[name] = (int(round(H)), int(round(W)))

        return cls(
            names=document.names,
            ks=ks,
            shapes=shapes,
        )


@dataclass(frozen=True, slots=True)
class _OpenCVExtrinsics:
    names: list[str]
    rots: dict[str, np.ndarray]
    ts: dict[str, np.ndarray]

    @classmethod
    def read(cls, path: str | Path) -> "_OpenCVExtrinsics":
        document = _read_opencv_yaml(path)

        rots: dict[str, np.ndarray] = {}
        ts: dict[str, np.ndarray] = {}

        for name in document.names:
            R = document.matrices.get(f"Rot_{name}")
            T = document.matrices.get(f"T_{name}")

            if R is None or T is None:
                raise ValueError(
                    f"missing Rot/T calibration for camera {name}"
                )
            if R.shape != (3, 3):
                raise ValueError(
                    f"Rot_{name} must be 3x3, got {R.shape}"
                )
            if T.size != 3:
                raise ValueError(
                    f"T_{name} must contain 3 values, got {T.shape}"
                )

            rots[name] = R.astype(np.float64, copy=True)
            ts[name] = T.reshape(3).astype(np.float64, copy=True)

        return cls(
            names=document.names,
            rots=rots,
            ts=ts,
        )


@dataclass(frozen=True, slots=True)
class _OpenCVYamlDocument:
    names: list[str]
    matrices: dict[str, np.ndarray]
    scalars: dict[str, float]


def _read_opencv_yaml(path: str | Path) -> _OpenCVYamlDocument:
    """Parse the small OpenCV-YAML subset used by intri.yml / extri.yml.

    This deliberately avoids an OpenCV dependency. Supported constructs:
      names:
        - "0000"
        - "0001"

      K_0000: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [...]

      H_0000: 1080.0

    It is not intended to be a general YAML parser.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines()

    names: list[str] = []
    matrices: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped or stripped.startswith("%YAML") or stripped == "---":
            i += 1
            continue

        if stripped == "names:":
            i += 1
            while i < len(lines):
                item = lines[i].strip()
                if not item.startswith("-"):
                    break

                raw = item[1:].strip()
                try:
                    name = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    name = raw.strip("\"'")

                names.append(str(name))
                i += 1
            continue

        matrix_match = re.fullmatch(
            r"([A-Za-z0-9_]+):\s*!!opencv-matrix",
            stripped,
        )
        if matrix_match:
            key = matrix_match.group(1)

            rows = None
            cols = None
            data_text = None

            i += 1
            while i < len(lines):
                child = lines[i]
                if child and not child[0].isspace():
                    break

                child_stripped = child.strip()

                if child_stripped.startswith("rows:"):
                    rows = int(child_stripped.split(":", 1)[1].strip())
                elif child_stripped.startswith("cols:"):
                    cols = int(child_stripped.split(":", 1)[1].strip())
                elif child_stripped.startswith("data:"):
                    data_text = child_stripped.split(":", 1)[1].strip()

                    # Support a bracketed data list continuing across lines.
                    while data_text.count("[") > data_text.count("]"):
                        i += 1
                        if i >= len(lines):
                            raise ValueError(
                                f"unterminated data list for {key} in {path}"
                            )
                        data_text += " " + lines[i].strip()

                i += 1

            if rows is None or cols is None or data_text is None:
                raise ValueError(
                    f"incomplete opencv-matrix {key} in {path}"
                )

            try:
                values = ast.literal_eval(data_text)
            except (ValueError, SyntaxError) as error:
                raise ValueError(
                    f"cannot parse matrix data for {key} in {path}"
                ) from error

            array = np.asarray(values, dtype=np.float64)
            if array.size != rows * cols:
                raise ValueError(
                    f"{key}: expected {rows * cols} values, "
                    f"got {array.size}"
                )

            matrices[key] = array.reshape(rows, cols)
            continue

        scalar_match = re.fullmatch(
            r"([A-Za-z0-9_]+):\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            stripped,
        )
        if scalar_match:
            scalars[scalar_match.group(1)] = float(scalar_match.group(2))
            i += 1
            continue

        i += 1

    if not names:
        raise ValueError(f"OpenCV calibration file has no camera names: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"camera names are not unique in {path}")

    return _OpenCVYamlDocument(
        names=names,
        matrices=matrices,
        scalars=scalars,
    )


def _processed_intrinsics(
    intri: _OpenCVIntrinsics,
    *,
    camera_name: str,
    scale: float,
) -> tuple[np.ndarray, int, int]:
    """Prepare K for initial viewer FOV without OpenCV.

    For the interactive viewer, calibration is used only to recover the initial
    aspect ratio and FOV. Distortion correction/cropping belongs to image
    preprocessing and is not needed to determine the calibrated camera pose.

    `data.scale` is applied consistently to both image dimensions and K.
    """

    K = intri.ks[camera_name].copy()
    H, W = intri.shapes[camera_name]

    width = max(1, int(round(W * scale)))
    height = max(1, int(round(H * scale)))

    K[0, :] *= scale
    K[1, :] *= scale
    K[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return K, width, height

def _validate_checkpoint(source: Any) -> None:
    if not isinstance(source, _FreeTimeGSGaussians):
        raise TypeError(
            "checkpoint root is not ftgspp.models.gaussians.Gaussians"
        )

    required = (
        "means",
        "scales",
        "quats",
        "opacities",
        "sh_0",
        "sh_n",
        "times",
        "durations",
        "velocity_model",
        "marginal_gates",
        "sh_degree",
        "max_duration",
    )
    missing = [name for name in required if not hasattr(source, name)]
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")

    if getattr(source, "_modules", None):
        raise NotImplementedError(
            "this importer currently supports explicit per-Gaussian velocity only"
        )

    _require_shape("means", source.means, (None, 3))
    n = int(source.means.shape[0])
    _require_shape("scales", source.scales, (n, 3))
    _require_shape("quats", source.quats, (n, 4))
    _require_scalar("opacities", source.opacities, n)
    _require_shape("sh_0", source.sh_0, (n, 1, 3))
    _require_sh_n(source.sh_n, n)
    _require_scalar("times", source.times, n)
    _require_scalar("durations", source.durations, n)
    _require_shape("velocity_model", source.velocity_model, (n, 3))
    _require_scalar("marginal_gates", source.marginal_gates, n)

    sh_degree = int(source.sh_degree)
    expected = (sh_degree + 1) ** 2 - 1
    if source.sh_n.shape[1] != expected:
        raise ValueError(
            f"sh_degree={sh_degree} requires {expected} sh_n coefficients, "
            f"got {source.sh_n.shape[1]}"
        )

    for name in (
        "means",
        "scales",
        "quats",
        "opacities",
        "sh_0",
        "sh_n",
        "times",
        "durations",
        "velocity_model",
        "marginal_gates",
    ):
        tensor = getattr(source, name)
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating-point")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinity")


def _require_shape(
    name: str,
    value: Any,
    shape: tuple[int | None, ...],
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != len(shape):
        raise ValueError(
            f"{name} must have {len(shape)} dimensions, got {value.ndim}"
        )

    for axis, expected in enumerate(shape):
        if expected is not None and value.shape[axis] != expected:
            raise ValueError(
                f"{name} axis {axis} must be {expected}, got {value.shape[axis]}"
            )


def _require_scalar(name: str, value: Any, n: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) not in {(n,), (n, 1)}:
        raise ValueError(
            f"{name} must have shape [N] or [N,1], got {tuple(value.shape)}"
        )


def _require_sh_n(value: Any, n: int) -> None:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 3
        or value.shape[0] != n
        or value.shape[2] != 3
    ):
        raise ValueError(
            f"sh_n must have shape [N,K,3], got {getattr(value, 'shape', None)}"
        )