from __future__ import annotations

import torch

from viewer4d.core.model import GaussianFrame, Viewer4DGS, register_evaluator

REPRESENTATION = "freetimegs.linear_temporal_gaussian.v1"


@register_evaluator(REPRESENTATION)
def evaluate_freetimegs(model: Viewer4DGS, time: float) -> GaussianFrame:
    """Evaluate the portable FreeTimeGS representation at normalized time."""

    means = model.tensor("base.means")
    log_scales = model.tensor("base.log_scales")
    quats = model.tensor("base.quats")
    opacity_logits = model.tensor("base.opacity_logits")
    sh0 = model.tensor("appearance.sh0")
    shN = model.tensor("appearance.shN")

    canonical_times = model.tensor("motion.canonical_times")
    log_durations = model.tensor("motion.log_durations")
    velocities = model.tensor("motion.velocities")

    options = model.metadata.get("representation_options", {})
    if not isinstance(options, dict):
        raise TypeError("metadata.representation_options must be a mapping")
    min_duration = float(options.get("min_duration", 0.02))
    min_opacity = float(options.get("min_opacity", 1e-4))

    t = torch.as_tensor(time, device=means.device, dtype=means.dtype)
    means_t = means + velocities * (t - canonical_times)

    durations = torch.exp(log_durations).clamp_min(min_duration)
    temporal_opacity = torch.exp(
        -0.5 * ((t - canonical_times) / (durations + 1e-8)).square()
    )
    opacities_t = torch.sigmoid(opacity_logits) * temporal_opacity
    opacities_t = opacities_t.clamp_min(min_opacity).squeeze(-1)

    return GaussianFrame(
        means=means_t,
        quats=quats,
        scales=torch.exp(log_scales),
        opacities=opacities_t,
        sh_coeffs=torch.cat((sh0, shN), dim=1),
    )


def create_freetimegs_model(
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    opacity_logits: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    canonical_times: torch.Tensor,
    log_durations: torch.Tensor,
    velocities: torch.Tensor,
    num_frames: int,
    fps: float,
    extras: dict[str, torch.Tensor] | None = None,
    source_metadata: dict | None = None,
) -> Viewer4DGS:
    """Build a portable Viewer4DGS from already-extracted FreeTimeGS tensors."""

    from viewer4d.core.model import SequenceInfo

    tensors: dict = {
        "base": {
            "means": means,
            "log_scales": log_scales,
            "quats": quats,
            "opacity_logits": opacity_logits,
        },
        "appearance": {
            "sh0": sh0,
            "shN": shN,
        },
        "motion": {
            "canonical_times": canonical_times,
            "log_durations": log_durations,
            "velocities": velocities,
        },
    }
    if extras:
        tensors["extras"] = extras

    return Viewer4DGS(
        representation=REPRESENTATION,
        sequence=SequenceInfo(num_frames=num_frames, fps=fps),
        tensors=tensors,
        metadata={
            "source": source_metadata or {},
            "representation_options": {
                "min_duration": 0.02,
                "min_opacity": 1e-4,
            },
        },
    )
