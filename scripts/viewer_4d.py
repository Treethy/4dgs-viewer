'''
    uv run python scripts/viewer_4d.py \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/_run/selfcap_ftgs/bike1/00/gaussians.pt \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/_run/selfcap_ftgs/bike1/00/config.toml \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/data/selfcap/bike/optimized/intri.yml \
        /gs/bs/tga-RLA/xu/FreeTimeGSPlusPlus/data/selfcap/bike/optimized/extri.yml \
        --port 8080
'''



from __future__ import annotations

import argparse
from pathlib import Path

import torch

from viewer4d.conversion.freetimegs import (
    FreeTimeGSImporter,
    load_freetimegs_eval_camera,
)
from viewer4d.visualization import Gaussian4DViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive 4D viewer for a FreeTimeGS++ checkpoint. "
            "Frame, playback speed, loop, and render mode are controlled "
            "from the browser GUI."
        )
    )

    parser.add_argument("checkpoint", type=Path, help="FreeTimeGS++ gaussians.pt")
    parser.add_argument("config", type=Path, help="Matching FTGS++ TOML config")
    parser.add_argument("intrinsics", type=Path, help="OpenCV intri.yml")
    parser.add_argument("extrinsics", type=Path, help="OpenCV extri.yml")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Maximum gsplat render width. Browser aspect ratio is preserved.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--show-axes", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print("[1/3] loading FreeTimeGS++ and converting to AnytimeGS")
    model = FreeTimeGSImporter().convert(
        args.checkpoint,
        config=args.config,
        device=args.device,
    )

    print("[2/3] loading first eval camera")
    initial_camera = load_freetimegs_eval_camera(
        config=args.config,
        intrinsics=args.intrinsics,
        extrinsics=args.extrinsics,
    )

    print(f"  eval camera: {initial_camera.name}")
    print(f"  frames:      {model.sequence.num_frames}")
    print(f"  source FPS:  {model.sequence.fps:g}")
    print(f"  Gaussians:   {model.num_gaussians:,}")

    print("[3/3] starting 4D viewer")
    print(f"  server: http://{args.host}:{args.port}")
    print("  Playback speed is controlled in the browser GUI.")

    viewer = Gaussian4DViewer(
        model,
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
