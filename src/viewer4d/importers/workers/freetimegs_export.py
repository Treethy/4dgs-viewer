from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_EXTRACT_FORMAT = "viewer4d.freetimegs_extracted"
_EXTRACT_VERSION = 1

_REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "means": ("means",),
    "log_scales": ("scales", "log_scales"),
    "quats": ("quats",),
    "opacity_logits": ("opacities", "opacity_logits"),
    "sh0": ("sh_0", "sh0"),
    "shN": ("sh_n", "shN"),
    "canonical_times": ("times", "canonical_times"),
    "log_durations": ("durations", "log_durations"),
    "velocities": ("velocity_model", "velocities"),
}

_OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "marginal_gates": ("marginal_gates",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Internal viewer4d worker: load a trusted FreeTimeGS checkpoint in "
            "its source environment and export a weights-only-safe tensor payload."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _select_model_root(root: Any) -> Any:
    """Find the object or mapping that actually contains Gaussian parameters."""

    if hasattr(root, "state_dict"):
        return root

    if isinstance(root, Mapping):
        # Common checkpoint wrappers used by Gaussian projects.
        for key in ("gaussians", "splats", "model"):
            if key in root:
                candidate = root[key]
                if hasattr(candidate, "state_dict") or isinstance(candidate, Mapping):
                    return candidate
        return root

    raise TypeError(
        "Unsupported FreeTimeGS checkpoint root type: "
        f"{type(root).__module__}.{type(root).__qualname__}"
    )


def _to_state_mapping(model_root: Any) -> Mapping[str, Any]:
    if hasattr(model_root, "state_dict"):
        state = model_root.state_dict()
    elif isinstance(model_root, Mapping):
        state = model_root
    else:
        raise TypeError(
            "Gaussian container must expose state_dict() or be a mapping, got "
            f"{type(model_root).__name__}"
        )

    if not isinstance(state, Mapping):
        raise TypeError("state_dict() did not return a mapping")
    return state


def _find_tensor(
    state: Mapping[str, Any],
    aliases: tuple[str, ...],
    *,
    required: bool,
) -> Tensor | None:
    # Prefer exact keys.
    for alias in aliases:
        value = state.get(alias)
        if isinstance(value, Tensor):
            return value

    # Also support prefixed state_dict keys such as "splats.means".
    matches: list[tuple[str, Tensor]] = []
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        if any(key.endswith(f".{alias}") for alias in aliases):
            matches.append((key, value))

    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        names = ", ".join(key for key, _ in matches)
        raise KeyError(
            f"Ambiguous state_dict matches for aliases {aliases}: {names}"
        )

    if required:
        available = ", ".join(str(key) for key in state.keys())
        raise KeyError(
            f"Missing required FreeTimeGS tensor; tried aliases {aliases}. "
            f"Available keys: {available}"
        )
    return None


def _cpu_tensor(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu").contiguous()


def _primitive_metadata(model_root: Any, source: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "type": "freetimegs",
        "source_file": str(source.resolve()),
        "source_class": (
            f"{type(model_root).__module__}.{type(model_root).__qualname__}"
        ),
    }

    for name in ("sh_degree", "max_duration"):
        if not hasattr(model_root, name):
            continue
        value = getattr(model_root, name)
        if isinstance(value, (bool, int, float, str)):
            # Keep non-finite values printable and portable across readers.
            if isinstance(value, float) and not math.isfinite(value):
                metadata[name] = str(value)
            else:
                metadata[name] = value
    return metadata


def main() -> None:
    args = _parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    # This worker is intentionally run only after the caller explicitly trusts
    # the source checkpoint. Loading arbitrary pickle files is unsafe.
    loaded = torch.load(source, map_location="cpu", weights_only=False)
    model_root = _select_model_root(loaded)
    state = _to_state_mapping(model_root)

    tensors: dict[str, Tensor] = {}
    for canonical_name, aliases in _REQUIRED_ALIASES.items():
        tensor = _find_tensor(state, aliases, required=True)
        assert tensor is not None
        tensors[canonical_name] = _cpu_tensor(tensor)

    extras: dict[str, Tensor] = {}
    for canonical_name, aliases in _OPTIONAL_ALIASES.items():
        tensor = _find_tensor(state, aliases, required=False)
        if tensor is not None:
            extras[canonical_name] = _cpu_tensor(tensor)

    payload: dict[str, Any] = {
        "format": _EXTRACT_FORMAT,
        "schema_version": _EXTRACT_VERSION,
        "tensors": tensors,
        "extras": extras,
        "source_metadata": _primitive_metadata(model_root, source),
    }
    torch.save(payload, output)


if __name__ == "__main__":
    main()
