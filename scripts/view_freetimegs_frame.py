'''
    uv run python scripts/view_freetimegs_frame.py \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/_run/selfcap_ftgs/bike1/00/gaussians.pt \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/_run/selfcap_ftgs/bike1/00/config.toml \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/data/selfcap/bike/optimized/intri.yml \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/data/selfcap/bike/optimized/extri.yml \
        0.5 \
        --port 8080
'''


from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from viewer4d.conversion.freetimegs import (
    FreeTimeGSImporter,
    load_freetimegs_eval_camera,
)
from viewer4d.visualization import GaussianViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one normalized-time GaussianFrame from a FreeTimeGS++ "
            "checkpoint using the first eval camera as the initial Viser view."
        )
    )

    parser.add_argument("checkpoint", type=Path, help="FreeTimeGS++ gaussians.pt")
    parser.add_argument("config", type=Path, help="Matching FTGS++ TOML config")
    parser.add_argument("intrinsics", type=Path, help="OpenCV intri.yml")
    parser.add_argument("extrinsics", type=Path, help="OpenCV extri.yml")
    parser.add_argument("time", type=float, help="Normalized time in [0,1]")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--show-axes", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.time <= 1.0:
        raise ValueError(f"time must be in [0,1], got {args.time}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print("[1/4] loading FreeTimeGS++ checkpoint and converting to AnytimeGS")
    model = FreeTimeGSImporter().convert(
        args.checkpoint,
        config=args.config,
        device=args.device,
    )

    print("[2/4] loading first eval camera from TOML + intri.yml + extri.yml")
    initial_camera = load_freetimegs_eval_camera(
        config=args.config,
        intrinsics=args.intrinsics,
        extrinsics=args.extrinsics,
    )

    nearest_frame = model.sequence.time_to_frame(args.time)
    source_sequence = model.metadata.get("source_sequence", {})
    source_start = source_sequence.get("frame_start")

    print(f"  eval camera name: {initial_camera.name}")
    print(f"  calibrated size: {initial_camera.width}x{initial_camera.height}")
    print(f"  camera position: {initial_camera.position.round(6).tolist()}")
    print(f"  camera forward:  {initial_camera.forward.round(6).tolist()}")
    print(f"  camera up:       {initial_camera.up.round(6).tolist()}")
    print(f"  normalized time: {args.time:.6f}")
    print(f"  nearest frame:   {nearest_frame}")
    if source_start is not None:
        print(f"  source frame:    {int(source_start) + nearest_frame}")

    print("[3/4] evaluating static GaussianFrame")
    frame = model.at_time(args.time)

    # Keep only the render-ready static frame while the viewer is running.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Gaussians: {frame.num_gaussians:,}")

    print("[4/4] starting Viser")
    print(f"  server: http://{args.host}:{args.port}")
    print(
        "  SSH tunnel from your local machine, for example:\n"
        f"    ssh -N -L {args.port}:127.0.0.1:{args.port} <server>\n"
        f"  then open http://127.0.0.1:{args.port}"
    )

    viewer = GaussianViewer(
        frame,
        initial_camera=initial_camera,
        device=args.device,
        host=args.host,
        port=args.port,
        render_width=args.width,
        jpeg_quality=args.jpeg_quality,
        show_axes=args.show_axes,
    )
    viewer.run()


if __name__ == "__main__":
    main()