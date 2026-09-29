"""What the stylised body is, as data. No bpy.

Epic Fight's limb is THREE rigid pieces: an upper box, a forearm box, and four vertices of
the ring between them owned by Elbow_*/Knee_* at weight 1.0. 196 of the body's 260
vertices carry a single influence, and on the limbs 159 of 160 do, so linear blend
skinning has nothing to blend there. A fold swings the boxes apart and the marker's slide
fills the corner between them; the whole deformation lives in a 1.4 mm band.

This body cuts loops along each limb and replaces those single influences with a two-bone
blend across the joint, which is a shape the game does not have. It is a viewport aid
only, measured 0.043 to 0.196 m off Epic Fight's own skinning on shipped poses; the export
reads the 20 joint matrices and never looks at a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SmoothLimb", "SMOOTH_LIMBS", "BLEND_FRACTION", "blend_band", "lower_share",
           "SMOOTH_SUFFIX"]

SMOOTH_SUFFIX = ".smooth"

BLEND_FRACTION = 0.80


@dataclass(frozen=True)
class SmoothLimb:
    upper: str
    lower: str
    marker: str


SMOOTH_LIMBS = (
    SmoothLimb("Arm_R", "Hand_R", "Elbow_R"),
    SmoothLimb("Arm_L", "Hand_L", "Elbow_L"),
    SmoothLimb("Thigh_R", "Leg_R", "Knee_R"),
    SmoothLimb("Thigh_L", "Leg_L", "Knee_L"),
)


def blend_band(upper_length: float, lower_length: float) -> float:
    return BLEND_FRACTION * min(upper_length, lower_length)


def lower_share(distance: float, band: float) -> float:
    """How much of a vertex belongs to the lower bone, `distance` being metres along the
    limb from the joint. Smoothstep, because a straight ramp creases where it starts."""
    if band <= 0.0:
        return 1.0 if distance > 0.0 else 0.0
    t = 0.5 + 0.5 * distance / band
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)
