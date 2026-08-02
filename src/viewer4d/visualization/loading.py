from __future__ import annotations

from pathlib import Path
from typing import Literal

from viewer4d.core.model import Viewer4DGS
from viewer4d.importers.freetimegs import FreeTimeGSImporter

InputFormat = Literal["auto", "portable", "freetimegs"]


class ModelLoadError(RuntimeError):
    """Raised when a viewer model cannot be opened with the requested format."""


def load_viewer_model(
    path: str | Path,
    *,
    input_format: InputFormat = "auto",
    device: str = "cpu",
    source_project: str | Path | None = None,
    num_frames: int | None = None,
    fps: float | None = None,
    trust_source: bool = False,
    save_converted_to: str | Path | None = None,
    overwrite: bool = False,
    mmap: bool = False,
    uv_executable: str = "uv",
    no_sync: bool = True,
) -> Viewer4DGS:
    """Open either a portable ``.v4d.pt`` or an original FreeTimeGS file.

    ``auto`` first attempts the weights-only-safe portable format. It falls back
    to FreeTimeGS only when ``source_project`` is supplied. Loading an original
    checkpoint still requires ``trust_source=True`` because the source worker
    must use ``torch.load(weights_only=False)`` in the source project environment.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if input_format not in ("auto", "portable", "freetimegs"):
        raise ValueError(
            "input_format must be one of: auto, portable, freetimegs; "
            f"got {input_format!r}"
        )

    if input_format == "portable":
        return _load_portable(
            source,
            device=device,
            mmap=mmap,
            save_converted_to=save_converted_to,
            overwrite=overwrite,
        )

    if input_format == "freetimegs":
        return _load_freetimegs(
            source,
            device=device,
            source_project=source_project,
            num_frames=num_frames,
            fps=fps,
            trust_source=trust_source,
            save_converted_to=save_converted_to,
            overwrite=overwrite,
            uv_executable=uv_executable,
            no_sync=no_sync,
        )

    portable_error: Exception | None = None
    try:
        return _load_portable(
            source,
            device=device,
            mmap=mmap,
            save_converted_to=save_converted_to,
            overwrite=overwrite,
        )
    except Exception as error:  # PyTorch uses several exception types here.
        portable_error = error

    if source_project is None:
        raise ModelLoadError(
            f"{source} is not a valid viewer4d portable model. "
            "For an original FreeTimeGS checkpoint, pass input_format='freetimegs' "
            "and source_project=/path/to/FreeTimeGSPlusPlus."
        ) from portable_error

    try:
        return _load_freetimegs(
            source,
            device=device,
            source_project=source_project,
            num_frames=num_frames,
            fps=fps,
            trust_source=trust_source,
            save_converted_to=save_converted_to,
            overwrite=overwrite,
            uv_executable=uv_executable,
            no_sync=no_sync,
        )
    except Exception as freetimegs_error:
        raise ModelLoadError(
            f"Could not open {source} as either a portable viewer4d model or "
            "a FreeTimeGS checkpoint."
        ) from freetimegs_error


def _load_portable(
    source: Path,
    *,
    device: str,
    mmap: bool,
    save_converted_to: str | Path | None,
    overwrite: bool,
) -> Viewer4DGS:
    model = Viewer4DGS.load(source, device=device, mmap=mmap)
    if save_converted_to is not None:
        model.save(save_converted_to, overwrite=overwrite)
    return model


def _load_freetimegs(
    source: Path,
    *,
    device: str,
    source_project: str | Path | None,
    num_frames: int | None,
    fps: float | None,
    trust_source: bool,
    save_converted_to: str | Path | None,
    overwrite: bool,
    uv_executable: str,
    no_sync: bool,
) -> Viewer4DGS:
    if source_project is None:
        raise TypeError("source_project is required for an original FreeTimeGS file")
    if num_frames is None:
        raise TypeError("num_frames is required for an original FreeTimeGS file")
    if fps is None:
        raise TypeError("fps is required for an original FreeTimeGS file")

    importer = FreeTimeGSImporter(
        source_project=source_project,
        trust_source=trust_source,
        uv_executable=uv_executable,
        no_sync=no_sync,
    )
    return Viewer4DGS.import_source(
        source,
        importer=importer,
        save_converted_to=save_converted_to,
        overwrite=overwrite,
        device=device,
        num_frames=num_frames,
        fps=fps,
    )
