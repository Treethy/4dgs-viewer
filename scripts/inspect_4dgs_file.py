#!/usr/bin/env python3
"""Inspect the structure of PyTorch checkpoints and PLY files.

Examples:
    uv run python scripts/inspect_4dgs_file.py model.pt
    uv run python scripts/inspect_4dgs_file.py point_cloud.ply
    uv run python scripts/inspect_4dgs_file.py model.pt --max-depth 6

PyTorch files are loaded safely by default with weights_only=True. Only use
--unsafe-load for files you trust, because arbitrary pickle data may execute code.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin"}
PLY_SCALAR_SIZES = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def format_bytes(num_bytes: int | float) -> str:
    """Format a byte count using binary units."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} TiB"


def short_repr(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def inspect_torch_file(
    path: Path,
    *,
    max_depth: int,
    max_items: int,
    unsafe_load: bool,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to inspect .pt/.pth/.ckpt files. "
            "Run this script through the project environment: uv run python ..."
        ) from exc

    print(f"File: {path}")
    print(f"Type: PyTorch checkpoint")
    print(f"File size: {format_bytes(path.stat().st_size)}")
    print(f"Load mode: {'UNSAFE pickle load' if unsafe_load else 'safe weights-only load'}")
    print()

    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=not unsafe_load,
        )
    except Exception as exc:
        if not unsafe_load:
            print("Safe loading failed.", file=sys.stderr)
            print(
                "If this is your own trusted checkpoint and it contains custom Python "
                "objects, retry with --unsafe-load.",
                file=sys.stderr,
            )
        raise RuntimeError(f"Could not load checkpoint: {exc}") from exc

    print("Structure:")
    seen: set[int] = set()
    _print_object(
        obj,
        name="root",
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        seen=seen,
    )


def _print_object(
    obj: Any,
    *,
    name: str,
    depth: int,
    max_depth: int,
    max_items: int,
    seen: set[int],
) -> None:
    indent = "  " * depth
    prefix = f"{indent}- {name}: "

    # Import lazily so PLY inspection does not need these packages.
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    if torch is not None and isinstance(obj, torch.Tensor):
        shape = tuple(obj.shape)
        numel = obj.numel()
        nbytes = numel * obj.element_size()
        print(
            prefix
            + f"Tensor shape={shape}, dtype={obj.dtype}, device={obj.device}, "
            + f"numel={numel:,}, size={format_bytes(nbytes)}, "
            + f"requires_grad={obj.requires_grad}"
        )
        return

    if np is not None and isinstance(obj, np.ndarray):
        print(
            prefix
            + f"ndarray shape={obj.shape}, dtype={obj.dtype}, "
            + f"numel={obj.size:,}, size={format_bytes(obj.nbytes)}"
        )
        return

    if obj is None or isinstance(obj, (bool, int, float, str, bytes, Path)):
        print(prefix + f"{type(obj).__name__} = {short_repr(obj)}")
        return

    # Avoid infinite recursion for self-referential containers/objects.
    object_id = id(obj)
    if object_id in seen:
        print(prefix + f"{type(obj).__name__} <already visited>")
        return

    if isinstance(obj, Mapping):
        seen.add(object_id)
        print(prefix + f"{type(obj).__name__} len={len(obj)}")
        if depth >= max_depth:
            print(f"{indent}  ... maximum depth reached")
            return
        items = list(obj.items())
        for key, value in items[:max_items]:
            _print_object(
                value,
                name=str(key),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
        if len(items) > max_items:
            print(f"{indent}  ... {len(items) - max_items} more entries omitted")
        return

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        seen.add(object_id)
        fields = dataclasses.fields(obj)
        print(prefix + f"dataclass {type(obj).__module__}.{type(obj).__qualname__}")
        if depth >= max_depth:
            print(f"{indent}  ... maximum depth reached")
            return
        for field in fields[:max_items]:
            _print_object(
                getattr(obj, field.name),
                name=field.name,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
        if len(fields) > max_items:
            print(f"{indent}  ... {len(fields) - max_items} more fields omitted")
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        seen.add(object_id)
        print(prefix + f"{type(obj).__name__} len={len(obj)}")
        if depth >= max_depth:
            print(f"{indent}  ... maximum depth reached")
            return
        for index, value in enumerate(obj[:max_items]):
            _print_object(
                value,
                name=f"[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
        if len(obj) > max_items:
            print(f"{indent}  ... {len(obj) - max_items} more entries omitted")
        return

    if hasattr(obj, "__dict__"):
        seen.add(object_id)
        attrs = vars(obj)
        print(prefix + f"object {type(obj).__module__}.{type(obj).__qualname__}")
        if depth >= max_depth:
            print(f"{indent}  ... maximum depth reached")
            return
        for key, value in list(attrs.items())[:max_items]:
            _print_object(
                value,
                name=key,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
        if len(attrs) > max_items:
            print(f"{indent}  ... {len(attrs) - max_items} more attributes omitted")
        return

    print(prefix + f"{type(obj).__module__}.{type(obj).__qualname__} = {short_repr(obj)}")


def inspect_ply_file(path: Path) -> None:
    file_size = path.stat().st_size
    elements: list[dict[str, Any]] = []
    comments: list[str] = []
    obj_info: list[str] = []
    ply_format: str | None = None
    version: str | None = None
    current_element: dict[str, Any] | None = None

    with path.open("rb") as file:
        first = file.readline()
        if first.strip() != b"ply":
            raise RuntimeError("The file does not start with the PLY magic word 'ply'.")

        while True:
            raw_line = file.readline()
            if not raw_line:
                raise RuntimeError("Unexpected end of file before 'end_header'.")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise RuntimeError("PLY header is not valid ASCII.") from exc

            if not line:
                continue
            tokens = line.split()
            keyword = tokens[0]

            if keyword == "format" and len(tokens) >= 3:
                ply_format, version = tokens[1], tokens[2]
            elif keyword == "comment":
                comments.append(line[len("comment") :].strip())
            elif keyword == "obj_info":
                obj_info.append(line[len("obj_info") :].strip())
            elif keyword == "element" and len(tokens) == 3:
                current_element = {
                    "name": tokens[1],
                    "count": int(tokens[2]),
                    "properties": [],
                }
                elements.append(current_element)
            elif keyword == "property":
                if current_element is None:
                    raise RuntimeError("Found a PLY property before any element declaration.")
                if len(tokens) == 3:
                    current_element["properties"].append(
                        {
                            "kind": "scalar",
                            "dtype": tokens[1],
                            "name": tokens[2],
                        }
                    )
                elif len(tokens) == 5 and tokens[1] == "list":
                    current_element["properties"].append(
                        {
                            "kind": "list",
                            "count_dtype": tokens[2],
                            "item_dtype": tokens[3],
                            "name": tokens[4],
                        }
                    )
                else:
                    raise RuntimeError(f"Unsupported PLY property line: {line}")
            elif keyword == "end_header":
                header_size = file.tell()
                break

    print(f"File: {path}")
    print("Type: PLY")
    print(f"File size: {format_bytes(file_size)}")
    print(f"Format: {ply_format or 'unknown'}")
    print(f"PLY version: {version or 'unknown'}")
    print(f"Header size: {format_bytes(header_size)}")
    print(f"Payload size: {format_bytes(max(0, file_size - header_size))}")

    if comments:
        print("Comments:")
        for comment in comments:
            print(f"  - {comment}")
    if obj_info:
        print("Object info:")
        for value in obj_info:
            print(f"  - {value}")

    print("Elements:")
    for element in elements:
        name = element["name"]
        count = element["count"]
        properties = element["properties"]
        print(f"- {name}: count={count:,}, properties={len(properties)}")

        fixed_row_size = 0
        has_variable_property = False
        for prop in properties:
            if prop["kind"] == "scalar":
                dtype = prop["dtype"]
                dtype_size = PLY_SCALAR_SIZES.get(dtype)
                if dtype_size is None:
                    has_variable_property = True
                else:
                    fixed_row_size += dtype_size
                size_text = f", bytes/value={dtype_size}" if dtype_size else ""
                print(f"  - {prop['name']}: {dtype}{size_text}")
            else:
                has_variable_property = True
                print(
                    f"  - {prop['name']}: list "
                    f"count={prop['count_dtype']}, item={prop['item_dtype']}"
                )

        if not has_variable_property:
            estimated = count * fixed_row_size
            print(
                f"  fixed row size={fixed_row_size} B, "
                f"estimated element data={format_bytes(estimated)}"
            )
        else:
            print("  element data size is variable because it contains list/unknown properties")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print keys, tensor shapes, dtypes, and PLY elements/properties."
    )
    parser.add_argument("path", type=Path, help="Path to a .pt/.pth/.ckpt or .ply file")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum recursive depth for PyTorch checkpoints (default: 8)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum entries shown per container (default: 100)",
    )
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow torch.load(weights_only=False). Use only for trusted files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path: Path = args.path.expanduser().resolve()

    if not path.exists():
        print(f"Error: file does not exist: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"Error: path is not a file: {path}", file=sys.stderr)
        return 2

    suffix = path.suffix.lower()
    try:
        if suffix == ".ply":
            inspect_ply_file(path)
        elif suffix in PT_SUFFIXES:
            inspect_torch_file(
                path,
                max_depth=max(0, args.max_depth),
                max_items=max(1, args.max_items),
                unsafe_load=args.unsafe_load,
            )
        else:
            raise RuntimeError(
                f"Unsupported extension '{suffix}'. Supported: "
                ".pt, .pth, .ckpt, .bin, .ply"
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
