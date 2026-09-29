"""Unit quaternions (w, x, y, z) and 3-vectors as plain tuples. No bpy, no numpy.

The retarget runs a few hundred thousand of these per clip, so they stay small free
functions over tuples rather than a class; efb.matrix keeps its own for the Mat4 side and
the two agree on the (w, x, y, z) order and on R(q) acting on column vectors.
"""

from __future__ import annotations

import math

from .matrix import Mat4, quat_conjugate, quat_mul, quat_normalize

__all__ = ["IDENTITY", "mul", "conj", "normalize", "rotate", "from_rotvec",
           "from_axis_angle", "to_rows", "from_rows", "from_mat4", "between", "angle",
           "nlerp", "slerp", "align", "dot3", "cross3", "norm3", "unit3", "sub3",
           "add3", "scale3", "lerp3", "twist_about", "signed_angle"]

IDENTITY = (1.0, 0.0, 0.0, 0.0)

mul = quat_mul
conj = quat_conjugate
normalize = quat_normalize


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm3(a):
    return math.sqrt(dot3(a, a))


def unit3(a, fallback=(0.0, 1.0, 0.0)):
    n = norm3(a)
    return fallback if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale3(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def rotate(q, v):
    """R(q) @ v, without building the matrix."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def from_axis_angle(axis, radians):
    ax = unit3(axis)
    s = math.sin(radians * 0.5)
    return (math.cos(radians * 0.5), ax[0] * s, ax[1] * s, ax[2] * s)


def from_rotvec(v):
    """Axis-angle as one vector, the SMPL / SOMA pose parameter."""
    theta = norm3(v)
    if theta < 1e-12:
        return normalize((1.0, v[0] * 0.5, v[1] * 0.5, v[2] * 0.5))
    s = math.sin(theta * 0.5) / theta
    return (math.cos(theta * 0.5), v[0] * s, v[1] * s, v[2] * s)


def to_rows(q):
    m = Mat4.from_quaternion(normalize(q)).rows()
    return (m[0][:3], m[1][:3], m[2][:3])


def from_rows(r):
    """3x3 rotation, given as rows, to a unit quaternion."""
    return Mat4.from_rows(((r[0][0], r[0][1], r[0][2], 0.0),
                           (r[1][0], r[1][1], r[1][2], 0.0),
                           (r[2][0], r[2][1], r[2][2], 0.0),
                           (0.0, 0.0, 0.0, 1.0))).to_quaternion()


def from_mat4(m: Mat4):
    return m.to_quaternion()


def between(u, v):
    """The shortest rotation taking direction u onto direction v."""
    u, v = unit3(u), unit3(v)
    d = dot3(u, v)
    if d < -1.0 + 1e-9:
        axis = cross3(u, (1.0, 0.0, 0.0))
        if norm3(axis) < 1e-6:
            axis = cross3(u, (0.0, 1.0, 0.0))
        return from_axis_angle(axis, math.pi)
    c = cross3(u, v)
    return normalize((1.0 + d, c[0], c[1], c[2]))


def angle(a, b=IDENTITY):
    """Radians between two orientations. atan2 rather than acos of the dot product, which
    reads 0.16 degrees off two keys that differ only in their sixth decimal."""
    r = mul(conj(normalize(a)), normalize(b))
    return 2.0 * math.atan2(math.sqrt(r[1] * r[1] + r[2] * r[2] + r[3] * r[3]), abs(r[0]))


def align(q, ref):
    """q or -q, whichever sits on ref's side of the sphere."""
    return q if sum(x * y for x, y in zip(q, ref)) >= 0.0 else tuple(-v for v in q)


def nlerp(a, b, t):
    b = align(b, a)
    return normalize(tuple(x + (y - x) * t for x, y in zip(a, b)))


def slerp(a, b, t):
    b = align(b, a)
    d = min(1.0, sum(x * y for x, y in zip(a, b)))
    if d > 0.9995:
        return nlerp(a, b, t)
    th = math.acos(d)
    s = math.sin(th)
    fa, fb = math.sin((1.0 - t) * th) / s, math.sin(t * th) / s
    return tuple(x * fa + y * fb for x, y in zip(a, b))


def twist_about(q, axis):
    """The part of q that turns about `axis` (swing-twist split), as a quaternion."""
    ax = unit3(axis)
    p = dot3((q[1], q[2], q[3]), ax)
    t = (q[0], ax[0] * p, ax[1] * p, ax[2] * p)
    n = math.sqrt(sum(v * v for v in t))
    return IDENTITY if n < 1e-12 else tuple(v / n for v in t)


def signed_angle(a, b, axis):
    """Radians from a to b about `axis`, both first flattened onto the plane it is
    normal to."""
    ax = unit3(axis)
    pa = sub3(a, scale3(ax, dot3(a, ax)))
    pb = sub3(b, scale3(ax, dot3(b, ax)))
    return math.atan2(dot3(ax, cross3(pa, pb)), dot3(pa, pb))
