"""Row-major 4x4 matrices and quaternions. No bpy, no numpy."""

from __future__ import annotations

import math

__all__ = ["Mat4", "quat_mul", "quat_conjugate", "quat_normalize", "quat_slerp"]

_IDENT = (1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0)


class Mat4:
    """Immutable 4x4, stored row-major: m[4*r + c]. Multiplication is column-vector
    convention, so `a @ b` applies b first. Matches Blender's mathutils.Matrix layout,
    which is why Mat4.rows() can be fed straight to mathutils.Matrix()."""

    __slots__ = ("m",)

    def __init__(self, values=_IDENT):
        m = tuple(float(v) for v in values)
        if len(m) != 16:
            raise ValueError("Mat4 needs 16 values, got %d" % len(m))
        object.__setattr__(self, "m", m)

    def __setattr__(self, *a):
        raise AttributeError("Mat4 is immutable")


    @classmethod
    def identity(cls) -> "Mat4":
        return cls(_IDENT)

    @classmethod
    def from_flat(cls, flat) -> "Mat4":
        """flat is 16 numbers in row-major order (the Epic Fight json layout)."""
        return cls(flat)

    @classmethod
    def from_rows(cls, rows) -> "Mat4":
        out = []
        for r in rows:
            out.extend(r)
        return cls(out)

    @classmethod
    def from_columns(cls, cols) -> "Mat4":
        return cls.from_rows(cols).transposed()

    @classmethod
    def translation(cls, x, y, z) -> "Mat4":
        return cls((1.0, 0.0, 0.0, float(x),
                    0.0, 1.0, 0.0, float(y),
                    0.0, 0.0, 1.0, float(z),
                    0.0, 0.0, 0.0, 1.0))

    @classmethod
    def scaling(cls, x, y, z) -> "Mat4":
        return cls((float(x), 0.0, 0.0, 0.0,
                    0.0, float(y), 0.0, 0.0,
                    0.0, 0.0, float(z), 0.0,
                    0.0, 0.0, 0.0, 1.0))

    @classmethod
    def rotation_axis(cls, radians, axis) -> "Mat4":
        ax, ay, az = axis
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n == 0.0:
            raise ValueError("zero rotation axis")
        ax, ay, az = ax / n, ay / n, az / n
        c = math.cos(radians)
        s = math.sin(radians)
        k = 1.0 - c
        return cls((ax * ax * k + c,      ax * ay * k - az * s, ax * az * k + ay * s, 0.0,
                    ay * ax * k + az * s, ay * ay * k + c,      ay * az * k - ax * s, 0.0,
                    az * ax * k - ay * s, az * ay * k + ax * s, az * az * k + c,      0.0,
                    0.0, 0.0, 0.0, 1.0))

    @classmethod
    def rotation_x(cls, radians) -> "Mat4":
        return cls.rotation_axis(radians, (1.0, 0.0, 0.0))

    @classmethod
    def rotation_y(cls, radians) -> "Mat4":
        return cls.rotation_axis(radians, (0.0, 1.0, 0.0))

    @classmethod
    def rotation_z(cls, radians) -> "Mat4":
        return cls.rotation_axis(radians, (0.0, 0.0, 1.0))

    @classmethod
    def from_quaternion(cls, q) -> "Mat4":
        """q is (w, x, y, z), the Blender / Epic-Fight-json order. Returns the standard
        right-handed active rotation R(q)."""
        w, x, y, z = q
        xx, yy, zz = 2.0 * x * x, 2.0 * y * y, 2.0 * z * z
        xy, xz, yz = 2.0 * x * y, 2.0 * x * z, 2.0 * y * z
        xw, yw, zw = 2.0 * x * w, 2.0 * y * w, 2.0 * z * w
        return cls((1.0 - yy - zz, xy - zw,       xz + yw,       0.0,
                    xy + zw,       1.0 - xx - zz, yz - xw,       0.0,
                    xz - yw,       yz + xw,       1.0 - xx - yy, 0.0,
                    0.0, 0.0, 0.0, 1.0))

    @classmethod
    def compose(cls, loc, quat, scale) -> "Mat4":
        """T(loc) @ R(quat) @ S(scale); quat is (w, x, y, z)."""
        return (cls.translation(*loc) @ cls.from_quaternion(quat)
                @ cls.scaling(*scale))


    def rows(self):
        m = self.m
        return (m[0:4], m[4:8], m[8:12], m[12:16])

    def columns(self):
        m = self.m
        return tuple(tuple(m[4 * r + c] for r in range(4)) for c in range(4))

    def flat(self):
        """Row-major 16-tuple: exactly what goes into an Epic Fight json transform."""
        return self.m

    def __getitem__(self, rc):
        r, c = rc
        return self.m[4 * r + c]

    def __repr__(self):
        return "Mat4(%s)" % ", ".join("[%s]" % " ".join("%.6g" % v for v in r)
                                      for r in self.rows())

    def __eq__(self, other):
        return isinstance(other, Mat4) and self.m == other.m

    def __hash__(self):
        return hash(self.m)


    def __matmul__(self, other: "Mat4") -> "Mat4":
        a, b = self.m, other.m
        out = [0.0] * 16
        for r in range(4):
            a0, a1, a2, a3 = a[4 * r], a[4 * r + 1], a[4 * r + 2], a[4 * r + 3]
            for c in range(4):
                out[4 * r + c] = (a0 * b[c] + a1 * b[4 + c]
                                  + a2 * b[8 + c] + a3 * b[12 + c])
        return Mat4(out)

    def transposed(self) -> "Mat4":
        m = self.m
        return Mat4(tuple(m[4 * c + r] for r in range(4) for c in range(4)))

    def inverse(self) -> "Mat4":
        """Gauss-Jordan with partial pivoting. Raises on a singular matrix."""
        a = [list(self.m[4 * r:4 * r + 4]) + [1.0 if i == r else 0.0 for i in range(4)]
             for r in range(4)]
        for col in range(4):
            piv = max(range(col, 4), key=lambda r: abs(a[r][col]))
            if abs(a[piv][col]) < 1e-20:
                raise ZeroDivisionError("singular matrix: %r" % (self,))
            a[col], a[piv] = a[piv], a[col]
            inv = 1.0 / a[col][col]
            a[col] = [v * inv for v in a[col]]
            for r in range(4):
                if r == col:
                    continue
                f = a[r][col]
                if f:
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
        return Mat4([v for r in a for v in r[4:]])


    def to_translation(self):
        m = self.m
        return (m[3], m[7], m[11])

    def to_scale(self):
        """Length of each basis column, i.e. Blender's decompose() scale (sign ignored)."""
        m = self.m
        return tuple(math.sqrt(m[c] ** 2 + m[4 + c] ** 2 + m[8 + c] ** 2)
                     for c in range(3))

    def to_quaternion(self):
        """(w, x, y, z) of the rotation part, columns normalised first."""
        m = self.m
        sx, sy, sz = self.to_scale()
        if sx == 0.0 or sy == 0.0 or sz == 0.0:
            return (1.0, 0.0, 0.0, 0.0)
        r = [m[0] / sx, m[1] / sy, m[2] / sz,
             m[4] / sx, m[5] / sy, m[6] / sz,
             m[8] / sx, m[9] / sy, m[10] / sz]
        tr = r[0] + r[4] + r[8]
        if tr >= 0.0:
            t = math.sqrt(tr + 1.0)
            w = 0.5 * t
            t = 0.5 / t
            x = (r[7] - r[5]) * t
            y = (r[2] - r[6]) * t
            z = (r[3] - r[1]) * t
        elif r[0] >= r[4] and r[0] >= r[8]:
            t = math.sqrt(r[0] - (r[4] + r[8]) + 1.0)
            x = 0.5 * t
            t = 0.5 / t
            y = (r[3] + r[1]) * t
            z = (r[2] + r[6]) * t
            w = (r[7] - r[5]) * t
        elif r[4] > r[8]:
            t = math.sqrt(r[4] - (r[8] + r[0]) + 1.0)
            y = 0.5 * t
            t = 0.5 / t
            z = (r[7] + r[5]) * t
            x = (r[3] + r[1]) * t
            w = (r[2] - r[6]) * t
        else:
            t = math.sqrt(r[8] - (r[0] + r[4]) + 1.0)
            z = 0.5 * t
            t = 0.5 / t
            x = (r[2] + r[6]) * t
            y = (r[7] + r[5]) * t
            w = (r[3] - r[1]) * t
        return quat_normalize((w, x, y, z))

    def decompose(self):
        """(loc, quat_wxyz, scale) such that Mat4.compose(*result) == self."""
        return (self.to_translation(), self.to_quaternion(), self.to_scale())


    def transform_point(self, p):
        m = self.m
        x, y, z = p
        return (m[0] * x + m[1] * y + m[2] * z + m[3],
                m[4] * x + m[5] * y + m[6] * z + m[7],
                m[8] * x + m[9] * y + m[10] * z + m[11])

    def transform_direction(self, v):
        m = self.m
        x, y, z = v
        return (m[0] * x + m[1] * y + m[2] * z,
                m[4] * x + m[5] * y + m[6] * z,
                m[8] * x + m[9] * y + m[10] * z)


    def max_abs_diff(self, other: "Mat4") -> float:
        return max(abs(a - b) for a, b in zip(self.m, other.m))

    def approx_equal(self, other: "Mat4", tol: float = 1e-6) -> bool:
        return self.max_abs_diff(other) <= tol


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_slerp(a, b, t):
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-v for v in b)
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(tuple(x + (y - x) * t for x, y in zip(a, b)))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    st = math.sin(theta)
    f0 = math.sin((1.0 - t) * theta) / st
    f1 = math.sin(t * theta) / st
    return tuple(x * f0 + y * f1 for x, y in zip(a, b))
