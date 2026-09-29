"""Line-for-line port of Epic Fight's OpenMatrix4f / JointTransform / JsonAssetLoader math.

This module exists to be the oracle: tests assert that the clean row-major math in
efb.matrix and efb.coords agrees with it on every matrix in the jar. Nothing else
should import it.

Source: epicfight-21.17.3.1.jar
  yesman/epicfight/api/utils/math/OpenMatrix4f
  yesman/epicfight/api/animation/JointTransform
  yesman/epicfight/api/asset/JsonAssetLoader

Storage is Epic Fight's: fields are named m<col><row> and hold logical[row][col],
i.e. column-major, translation in m30/m31/m32.
"""

from __future__ import annotations

import math

from .matrix import Mat4

__all__ = ["OpenMatrix4f", "joint_transform_to_matrix", "ef_load_joint_local",
           "ef_load_matrix_keyframe", "ef_load_attributes_keyframe",
           "BLENDER_TO_MINECRAFT_COORD", "MINECRAFT_TO_BLENDER_COORD"]

_ORDER = [(c, r) for c in range(4) for r in range(4)]


class OpenMatrix4f:
    __slots__ = ("v",)

    def __init__(self, v=None):
        if v is None:
            v = [1.0 if c == r else 0.0 for c, r in _ORDER]
        self.v = list(float(x) for x in v)

    def get(self, col, row):
        return self.v[col * 4 + row]

    def set(self, col, row, value):
        self.v[col * 4 + row] = value

    def copy(self):
        return OpenMatrix4f(self.v)

    @classmethod
    def load_flat(cls, elements):
        if len(elements) != 16:
            raise ValueError("need 16 elements")
        return cls(elements)

    def transpose(self):
        n = [0.0] * 16
        for c in range(4):
            for r in range(4):
                n[c * 4 + r] = self.v[r * 4 + c]
        self.v = n
        return self

    @staticmethod
    def mul(left, right):
        out = OpenMatrix4f()
        for c in range(4):
            for r in range(4):
                out.v[c * 4 + r] = sum(left.get(k, r) * right.get(c, k) for k in range(4))
        return out

    def mul_front(self, other):
        """this = other * this"""
        self.v = OpenMatrix4f.mul(other, self).v
        return self

    def mul_back(self, other):
        """this = this * other"""
        self.v = OpenMatrix4f.mul(self, other).v
        return self

    def determinant(self):
        return self.to_mat4_det()

    def to_mat4_det(self):
        m = self.to_mat4().m
        def d3(t00, t01, t02, t10, t11, t12, t20, t21, t22):
            return (t00 * (t11 * t22 - t12 * t21)
                    + t01 * (t12 * t20 - t10 * t22)
                    + t02 * (t10 * t21 - t11 * t20))
        g = self.get
        return (g(0, 0) * d3(g(1, 1), g(1, 2), g(1, 3), g(2, 1), g(2, 2), g(2, 3), g(3, 1), g(3, 2), g(3, 3))
                - g(1, 0) * d3(g(0, 1), g(0, 2), g(0, 3), g(2, 1), g(2, 2), g(2, 3), g(3, 1), g(3, 2), g(3, 3))
                + g(2, 0) * d3(g(0, 1), g(0, 2), g(0, 3), g(1, 1), g(1, 2), g(1, 3), g(3, 1), g(3, 2), g(3, 3))
                - g(3, 0) * d3(g(0, 1), g(0, 2), g(0, 3), g(1, 1), g(1, 2), g(1, 3), g(2, 1), g(2, 2), g(2, 3)))

    def invert(self):
        det = self.determinant()
        if det == 0.0:
            raise ZeroDivisionError("singular")
        inv = 1.0 / det

        def d3(t00, t01, t02, t10, t11, t12, t20, t21, t22):
            return (t00 * (t11 * t22 - t12 * t21)
                    + t01 * (t12 * t20 - t10 * t22)
                    + t02 * (t10 * t21 - t11 * t20))

        g = self.get
        t = {}
        cols = [0, 1, 2, 3]
        for c in range(4):
            for r in range(4):
                rc = [x for x in cols if x != c]
                rr = [x for x in cols if x != r]
                sub = [g(cc, rrr) for cc in rc for rrr in rr]
                t[(c, r)] = ((-1.0) ** (c + r)) * d3(*sub)
        out = OpenMatrix4f()
        for c in range(4):
            for r in range(4):
                out.set(c, r, t[(r, c)] * inv)
        return out

    @classmethod
    def create_rotator_deg(cls, degree, axis):
        angle = math.radians(degree)
        ax, ay, az = axis
        c = math.cos(angle)
        s = math.sin(angle)
        k = 1.0 - c
        xy, yz, xz = ax * ay, ay * az, ax * az
        xs, ys, zs = ax * s, ay * s, az * s
        out = cls()
        out.set(0, 0, ax * ax * k + c)
        out.set(0, 1, xy * k + zs)
        out.set(0, 2, xz * k - ys)
        out.set(1, 0, xy * k - zs)
        out.set(1, 1, ay * ay * k + c)
        out.set(1, 2, yz * k + xs)
        out.set(2, 0, xz * k + ys)
        out.set(2, 1, yz * k - xs)
        out.set(2, 2, az * az * k + c)
        return out

    @classmethod
    def from_quaternion(cls, q_xyzw):
        x, y, z, w = q_xyzw
        xy, xz, xw, yz, yw, zw = x * y, x * z, x * w, y * z, y * w, z * w
        xx, yy, zz = 2.0 * x * x, 2.0 * y * y, 2.0 * z * z
        out = cls()
        out.set(0, 0, 1.0 - yy - zz)
        out.set(0, 1, 2.0 * (xy - zw))
        out.set(0, 2, 2.0 * (xz + yw))
        out.set(1, 0, 2.0 * (xy + zw))
        out.set(1, 1, 1.0 - xx - zz)
        out.set(1, 2, 2.0 * (yz - xw))
        out.set(2, 0, 2.0 * (xz - yw))
        out.set(2, 1, 2.0 * (yz + xw))
        out.set(2, 2, 1.0 - xx - yy)
        return out

    def translate(self, vec):
        x, y, z = vec
        g = self.get
        self.set(3, 0, g(3, 0) + g(0, 0) * x + g(1, 0) * y + g(2, 0) * z)
        self.set(3, 1, g(3, 1) + g(0, 1) * x + g(1, 1) * y + g(2, 1) * z)
        self.set(3, 2, g(3, 2) + g(0, 2) * x + g(1, 2) * y + g(2, 2) * z)
        self.set(3, 3, g(3, 3) + g(0, 3) * x + g(1, 3) * y + g(2, 3) * z)
        return self

    def scale(self, vec):
        for c, s in enumerate(vec):
            for r in range(4):
                self.set(c, r, self.get(c, r) * s)
        return self

    def to_translation_vector(self):
        return (self.get(3, 0), self.get(3, 1), self.get(3, 2))

    def to_scale_vector(self):
        return tuple(math.sqrt(sum(self.get(c, r) ** 2 for r in range(3)))
                     for c in range(3))

    def to_quaternion(self):
        s = self.copy().transpose()
        g = s.get
        lens = [sum(g(c, r) ** 2 for r in range(3)) for c in range(3)]
        if any(l == 0.0 for l in lens):
            return (0.0, 0.0, 0.0, 1.0)
        for c in range(3):
            inv = 1.0 / math.sqrt(lens[c])
            for r in range(3):
                s.set(c, r, g(c, r) * inv)
        tr = g(0, 0) + g(1, 1) + g(2, 2)
        if tr >= 0.0:
            t = math.sqrt(tr + 1.0)
            w = t * 0.5
            t = 0.5 / t
            x = (g(1, 2) - g(2, 1)) * t
            y = (g(2, 0) - g(0, 2)) * t
            z = (g(0, 1) - g(1, 0)) * t
        elif g(0, 0) >= g(1, 1) and g(0, 0) >= g(2, 2):
            t = math.sqrt(g(0, 0) - (g(1, 1) + g(2, 2)) + 1.0)
            x = t * 0.5
            t = 0.5 / t
            y = (g(1, 0) + g(0, 1)) * t
            z = (g(0, 2) + g(2, 0)) * t
            w = (g(1, 2) - g(2, 1)) * t
        elif g(1, 1) > g(2, 2):
            t = math.sqrt(g(1, 1) - (g(2, 2) + g(0, 0)) + 1.0)
            y = t * 0.5
            t = 0.5 / t
            z = (g(2, 1) + g(1, 2)) * t
            x = (g(1, 0) + g(0, 1)) * t
            w = (g(2, 0) - g(0, 2)) * t
        else:
            t = math.sqrt(g(2, 2) - (g(0, 0) + g(1, 1)) + 1.0)
            z = t * 0.5
            t = 0.5 / t
            x = (g(0, 2) + g(2, 0)) * t
            y = (g(2, 1) + g(1, 2)) * t
            w = (g(0, 1) - g(1, 0)) * t
        return (x, y, z, w)

    def to_mat4(self) -> Mat4:
        return Mat4(tuple(self.get(c, r) for r in range(4) for c in range(4)))

    @classmethod
    def from_mat4(cls, mat: Mat4):
        out = cls()
        for r in range(4):
            for c in range(4):
                out.set(c, r, mat[r, c])
        return out


BLENDER_TO_MINECRAFT_COORD = OpenMatrix4f.create_rotator_deg(-90.0, (1.0, 0.0, 0.0))
MINECRAFT_TO_BLENDER_COORD = BLENDER_TO_MINECRAFT_COORD.invert()


def joint_transform_to_matrix(loc, quat_xyzw, sca) -> OpenMatrix4f:
    """JointTransform.toMatrix(): identity.translate(t).mulBack(fromQuaternion(q)).scale(s)"""
    m = OpenMatrix4f().translate(loc)
    m.mul_back(OpenMatrix4f.from_quaternion(quat_xyzw))
    m.scale(sca)
    return m


def _attributes_to_joint_transform(loc, rot, sca):
    """JsonAssetLoader: fromPrimitives(loc, -rot[1], -rot[2], -rot[3], rot[0], sca).
    json "rot" is [w, x, y, z]; Epic Fight feeds JOML the conjugate."""
    return joint_transform_to_matrix(loc, (-rot[1], -rot[2], -rot[3], rot[0]), sca)


def ef_load_joint_local(transform, is_root) -> OpenMatrix4f:
    """JsonAssetLoader.getJoint(): the Joint.localTransform Epic Fight ends up holding."""
    if isinstance(transform, dict):
        m = _attributes_to_joint_transform(transform["loc"], transform["rot"], transform["sca"])
    else:
        m = OpenMatrix4f.load_flat(transform).transpose()
    if is_root:
        m.mul_front(BLENDER_TO_MINECRAFT_COORD)
    return m


def ef_load_matrix_keyframe(flat, inv_local: OpenMatrix4f, root_correction: bool) -> OpenMatrix4f:
    """JsonAssetLoader.getTransformSheet(), MATRIX branch."""
    m = OpenMatrix4f.load_flat(flat).transpose()
    if root_correction:
        m.mul_front(BLENDER_TO_MINECRAFT_COORD)
    m.mul_front(inv_local)
    return m


def ef_load_attributes_keyframe(loc, rot, sca) -> OpenMatrix4f:
    """JsonAssetLoader.getTransformSheet(), ATTRIBUTES branch.
    Note upstream applies neither the root correction nor the inverse local here."""
    return _attributes_to_joint_transform(loc, rot, sca)
