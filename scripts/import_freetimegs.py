from __future__ import annotations

import argparse
from pathlib import Path

from viewer4d import FreeTimeGSImporter, Viewer4DGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a trusted FreeTimeGS/FreeTimeGS++ checkpoint into the "
            "viewer4d portable representation."
        )
    )
    parser.add_argument("input", type=Path, help="Original FreeTimeGS .pt file")
    parser.add_argument(
        "--source-project",
        type=Path,
        required=True,
        help="FreeTimeGS/FreeTimeGS++ uv project directory",
    )
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="OUTPUT_PT",
        help="Optionally save the converted portable model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --save path",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the returned in-memory model, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--trust-source",
        action="store_true",
        help="Required: confirm that the original pickle checkpoint is trusted",
    )
    parser.add_argument(
        "--sync-source-env",
        action="store_true",
        help="Allow uv to sync the source project before running the worker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importer = FreeTimeGSImporter(
        source_project=args.source_project,
        trust_source=args.trust_source,
        no_sync=not args.sync_source_env,
    )

    model = Viewer4DGS.import_source(
        args.input,
        importer=importer,
        save_converted_to=args.save,
        overwrite=args.overwrite,
        device=args.device,
        num_frames=args.num_frames,
        fps=args.fps,
    )

    means = model.tensor("base.means")
    print(f"representation: {model.representation}")
    print(f"gaussians: {means.shape[0]:,}")
    print(f"frames: {model.sequence.num_frames}")
    print(f"fps: {model.sequence.fps}")
    print(f"device: {model.device}")
    if args.save is not None:
        print(f"saved: {args.save.expanduser().resolve()}")
    else:
        print("saved: no (conversion existed only in this process)")


if __name__ == "__main__":
    main()
