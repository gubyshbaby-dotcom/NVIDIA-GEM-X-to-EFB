"""Blender camera -> VIX camera animation. No bpy.

VIX rebuilds the camera as

    world_pos = pos.yRot(radians(-yawLock - 90)) + entity.getPosition(pt)
    yaw = yawLock - ry     pitch = rx     roll = rz     fov = value

(CameraEventsFix.calculateFinalCameraPosition / TransformCam). Vec3.yRot is the
right-handed rotation about +Y, and Camera.setRotation builds
Ry(pi - yaw) Rx(-pitch) Rz(-roll) with forward (0,0,-1), up (0,1,0), left (-1,0,0).
Both halves therefore share the prefix Ry(pi - yawLock), which is exactly the entity
model -> world rotation (EF: Ry(180) Ry(-bodyYaw), same as vanilla). Strip it and what
is left is the entity's own frame:

    pos frame     x = forward, y = up, z = right   (blocks, origin at the entity feet)
    orientation   Ry(ry) Rx(-rx) Rz(-rz)           camera basis in entity model space

Blender's rig faces +Y and is Z-up, so a Blender-space offset b maps straight through:

    pos = (b.y, b.z, b.x)

The camera's local axes agree already - Blender looks down -Z with +Y up, so does
Minecraft's Camera - which leaves only the world basis to change:

    R_model = BLENDER_TO_MINECRAFT @ R_camera_relative_to_rig

Sign trap: ry is the NEGATED yaw offset (yaw = yawLock - ry) and rz is the negated
roll, while rx is the pitch as-is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .coords import BLENDER_TO_MINECRAFT
from .matrix import Mat4
from .vixcam import ChannelGroup, VixCameraAnimation, POS_CHANNELS, ROT_CHANNELS

__all__ = ["CameraKey", "RigFrame", "blender_offset_to_vix_pos",
           "vix_pos_to_blender_offset", "rot_to_vix_angles", "vix_angles_to_rot",
           "vertical_fov_degrees", "unwrap_degrees", "reduce_keys",
           "build_animation", "DEFAULT_TOLERANCE", "DEFAULT_DIGITS"]

DEFAULT_TOLERANCE = {"pos": 1e-4, "rot": 1e-3, "fov": 1e-3}
DEFAULT_DIGITS = 4
_EPS = 1e-9


@dataclass(frozen=True)
class CameraKey:
    time: float
    pos: tuple
    rot: tuple
    fov: float


def blender_offset_to_vix_pos(b):
    return (b[1], b[2], b[0])


def vix_pos_to_blender_offset(p):
    return (p[2], p[0], p[1])


def _orthonormal_rows(m: Mat4):
    """Gram-Schmidt on the 3x3 block, so a scaled or sheared object still decomposes."""
    cols = [[m[r, c] for r in range(3)] for c in range(3)]

    def norm(v):
        n = math.sqrt(sum(x * x for x in v))
        if n < _EPS:
            raise ValueError("degenerate rotation")
        return [x / n for x in v]

    x = norm(cols[0])
    d = sum(a * b for a, b in zip(cols[1], x))
    y = norm([a - d * b for a, b in zip(cols[1], x)])
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    return [[x[i], y[i], z[i]] for i in range(3)]


def _decompose_yxz(rows):
    """R = Ry(a) Rx(b) Rz(c) -> (a, b, c) in radians."""
    sb = max(-1.0, min(1.0, -rows[1][2]))
    b = math.asin(sb)
    if abs(rows[1][2]) < 1.0 - 1e-9:
        a = math.atan2(rows[0][2], rows[2][2])
        c = math.atan2(rows[1][0], rows[1][1])
    else:
        c = 0.0
        a = math.atan2(rows[0][1], rows[0][0]) if sb > 0 else math.atan2(-rows[0][1], rows[0][0])
    return a, b, c


def rot_to_vix_angles(rig_relative: Mat4):
    """Camera rotation relative to the rig (Blender basis) -> (rx, ry, rz) degrees."""
    model = BLENDER_TO_MINECRAFT @ Mat4.from_rows(
        [r + [0.0] for r in _orthonormal_rows(rig_relative)] + [[0.0, 0.0, 0.0, 1.0]])
    a, b, c = _decompose_yxz(model.rows())
    return (-math.degrees(b), math.degrees(a), -math.degrees(c))


def vix_angles_to_rot(rx, ry, rz) -> Mat4:
    """Inverse of rot_to_vix_angles: back to a Blender-basis rotation."""
    model = (Mat4.rotation_y(math.radians(ry))
             @ Mat4.rotation_x(math.radians(-rx))
             @ Mat4.rotation_z(math.radians(-rz)))
    return BLENDER_TO_MINECRAFT.inverse() @ model


@dataclass(frozen=True)
class RigFrame:
    """The frame VIX locks at SetAnim: the rig's yaw only, taken once at the shot start.

    Translation is not part of it - VIX adds the entity's live position every frame, so
    the caller passes the per-frame origin location to pos().
    """
    yaw: float
    residual: float

    @classmethod
    def from_matrix(cls, origin_world: Mat4) -> "RigFrame":
        rows = _orthonormal_rows(origin_world)
        fwd = (rows[0][1], rows[1][1], rows[2][1])
        yaw = math.atan2(-fwd[0], fwd[1])
        flat = _orthonormal_rows(Mat4.rotation_z(yaw))
        worst = 0.0
        for c in range(3):
            dot = sum(rows[r][c] * flat[r][c] for r in range(3))
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
        return cls(yaw, worst)

    @property
    def inverse(self) -> Mat4:
        return Mat4.rotation_z(-self.yaw)

    def pos(self, cam_loc, origin_loc):
        d = (cam_loc[0] - origin_loc[0], cam_loc[1] - origin_loc[1], cam_loc[2] - origin_loc[2])
        return blender_offset_to_vix_pos(self.inverse.transform_direction(d))

    def angles(self, cam_world: Mat4):
        return rot_to_vix_angles(self.inverse @ cam_world)


def vertical_fov_degrees(lens, sensor_x=36.0, sensor_y=24.0, sensor_fit="AUTO",
                         res_x=1920, res_y=1080, aspect_x=1.0, aspect_y=1.0):
    """Blender lens/sensor -> Minecraft's vertical fov in degrees.

    AUTO fits the sensor to whichever rendered dimension is larger and still measures it
    with sensor_x (BKE_camera_sensor_size only reaches for sensor_y under an explicit
    vertical fit), so the render resolution changes the answer.
    """
    if lens <= 0.0:
        raise ValueError("lens must be positive")
    xa, ya = res_x * aspect_x, res_y * aspect_y
    if sensor_fit == "VERTICAL":
        sensor, vertical = sensor_y, True
    elif sensor_fit == "HORIZONTAL":
        sensor, vertical = sensor_x, False
    else:
        sensor, vertical = sensor_x, ya > xa
    half = math.atan((sensor * 0.5) / lens)
    if not vertical:
        half = math.atan(math.tan(half) * ya / xa)
    return math.degrees(2.0 * half)


def unwrap_degrees(values):
    """Keep an angle track continuous - VIX lerps rx/ry/rz as plain floats."""
    out = list(values[:1])
    for v in values[1:]:
        out.append(v - 360.0 * round((v - out[-1]) / 360.0))
    return out


def reduce_keys(times, tracks, tol):
    """Douglas-Peucker over the channels of one group; returns the kept indices."""
    n = len(times)
    if tol is None or n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        span = times[hi] - times[lo]
        worst, at = -1.0, -1
        for i in range(lo + 1, hi):
            t = 0.0 if span <= _EPS else (times[i] - times[lo]) / span
            err = max(abs(tr[i] - (tr[lo] + (tr[hi] - tr[lo]) * t)) for tr in tracks)
            if err > worst:
                worst, at = err, i
        if worst > tol:
            keep[at] = True
            stack.append((lo, at))
            stack.append((at, hi))
    return [i for i, k in enumerate(keep) if k]


def _round(values, digits):
    return list(values) if digits is None else [round(v, digits) for v in values]


def _group(times, names, tracks, tol, digits):
    idx = reduce_keys(times, tracks, tol)
    t = _round([times[i] for i in idx], digits)
    if any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError("times collide after rounding to %r digits" % digits)
    return ChannelGroup(t, {n: _round([tr[i] for i in idx], digits)
                            for n, tr in zip(names, tracks)})


def build_animation(keys, tolerance=DEFAULT_TOLERANCE, digits=DEFAULT_DIGITS,
                    time_scale=None) -> VixCameraAnimation:
    """Turn sampled CameraKeys into a VIX animation, one time array per group."""
    if not keys:
        raise ValueError("no keys")
    tol = dict(tolerance or {})
    times = [k.time for k in keys]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("key times must be strictly increasing")
    pos = [[k.pos[i] for k in keys] for i in range(3)]
    rot = [unwrap_degrees([k.rot[i] for k in keys]) for i in range(3)]
    fov = [[k.fov for k in keys]]
    return VixCameraAnimation(
        pos=_group(times, POS_CHANNELS, pos, tol.get("pos"), digits),
        rot=_group(times, ROT_CHANNELS, rot, tol.get("rot"), digits),
        fov=_group(times, ("value",), fov, tol.get("fov"), digits),
        time_scale=time_scale,
    )
