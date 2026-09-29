"""Vector helpers and the IK pole angle. No bpy."""

from __future__ import annotations

import math

__all__ = ["add", "sub", "scale", "dot", "cross", "length", "normalize",
           "signed_angle", "pole_angle"]


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = length(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def signed_angle(u, v, axis):
    """Angle from u to v measured about axis, right-hand positive."""
    a = normalize(axis)
    return math.atan2(dot(cross(u, v), a), dot(u, v))


def pole_angle(root_head, root_tail, chain_tip, pole, root_x_axis):
    """Blender IK pole_angle that leaves a chain at rest when its pole is at `pole`.

    Blender rolls the chain about root_head -> chain_tip until the root bone's X axis
    lines up with the pole plane; pole_angle is the offset from that alignment. Sign is
    Blender's, which is the negative of the right-hand one.
    """
    root_dir = sub(root_tail, root_head)
    pole_normal = cross(sub(chain_tip, root_head), sub(pole, root_head))
    projected = cross(pole_normal, root_dir)
    return -signed_angle(root_x_axis, projected, root_dir)
