"""Epic Fight armature json (assets/epicfight/animmodels/entity/*.json)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .coords import BLENDER_TO_MINECRAFT
from .matrix import Mat4

__all__ = ["Joint", "Armature", "BIPED_GROUPS", "COORD_BONE", "ROOT_BONE"]

COORD_BONE = "Coord"
ROOT_BONE = "Root"


@dataclass(frozen=True)
class Joint:
    name: str
    index: int
    parent: str | None
    rest_local: Mat4
    children: tuple = ()

    @property
    def is_root(self) -> bool:
        return self.parent is None


class Armature:
    """Joints keyed by name, plus the index Epic Fight assigns them.

    Index rule, from JsonAssetLoader.loadArmature: walk the "joints" array in order,
    skip any entry named "Coord", number the rest from 0. The hierarchy tree is separate
    and does not have to agree with that order.
    """

    def __init__(self, name: str, joints, root: str):
        self.name = name
        self.joints = dict(joints)
        self.root = root
        self._by_index = {j.index: j for j in self.joints.values()}


    @classmethod
    def from_json(cls, obj: dict, name: str = "armature") -> "Armature":
        arm = obj["armature"] if "armature" in obj else obj
        fmt = arm.get("armature_format", "matrix")
        names = [n for n in arm["joints"] if n != COORD_BONE]
        index = {n: i for i, n in enumerate(names)}
        joints = {}

        def take(node, parent):
            jname = node["name"]
            if jname == COORD_BONE:
                kids = node.get("children", [])
                if len(kids) != 1:
                    raise ValueError("Coord bone must have exactly one child")
                return take(kids[0], parent)
            if jname not in index:
                raise ValueError("joint %r missing from the joints array" % jname)
            local = _read_transform(node["transform"], fmt)
            kids = tuple(c["name"] for c in node.get("children", []))
            joints[jname] = Joint(jname, index[jname], parent, local, kids)
            for c in node.get("children", []):
                take(c, jname)
            return jname

        root = take(arm["hierarchy"][0], None)
        missing = set(index) - set(joints)
        if missing:
            raise ValueError("joints declared but absent from the hierarchy: %s"
                             % sorted(missing))
        return cls(name, joints, root)

    @classmethod
    def from_file(cls, path) -> "Armature":
        with open(path, "rb") as fh:
            obj = json.loads(fh.read())
        return cls.from_json(obj, os.path.splitext(os.path.basename(str(path)))[0])

    @classmethod
    def from_bytes(cls, data: bytes, name: str = "armature") -> "Armature":
        return cls.from_json(json.loads(data), name)


    def to_json(self) -> dict:
        """Loader-compatible armature json. Note Epic Fight's own exporter
        (Armature.exportJoint) writes the key "identifier", which its loader cannot read -
        it reads "name". We write "name"."""
        def node(jname):
            j = self.joints[jname]
            out = {"name": j.name, "transform": list(j.rest_local.flat())}
            if j.children:
                out["children"] = [node(c) for c in j.children]
            return out

        return {"armature": {"joints": [j.name for j in self.ordered()],
                             "hierarchy": [node(self.root)]}}


    def __len__(self):
        return len(self.joints)

    def __contains__(self, name):
        return name in self.joints

    def __getitem__(self, name) -> Joint:
        return self.joints[name]

    def ordered(self):
        """Joints in Epic Fight index order."""
        return [self._by_index[i] for i in sorted(self._by_index)]

    def by_index(self, index: int) -> Joint:
        return self._by_index[index]

    def names(self):
        return [j.name for j in self.ordered()]

    def parent_of(self, name):
        p = self.joints[name].parent
        return self.joints[p] if p else None

    def chain_to_root(self, name):
        """[name, ..., root]"""
        out = []
        cur = name
        while cur is not None:
            out.append(cur)
            cur = self.joints[cur].parent
        return out

    def depth_first(self):
        """Joint names in hierarchy order, which is the order animation tracks use."""
        out = []

        def walk(n):
            out.append(n)
            for c in self.joints[n].children:
                walk(c)

        walk(self.root)
        return out


    def rest_model(self, name) -> Mat4:
        """Armature-space rest matrix, Blender frame. Blender's bone.matrix_local."""
        m = Mat4.identity()
        for n in reversed(self.chain_to_root(name)):
            m = m @ self.joints[n].rest_local
        return m

    def rest_model_minecraft(self, name) -> Mat4:
        """Armature-space rest matrix in Minecraft coords: B @ rest_model."""
        return BLENDER_TO_MINECRAFT @ self.rest_model(name)

    def ef_local(self, name) -> Mat4:
        """Joint.localTransform as Epic Fight builds it: B @ L for the root, L otherwise."""
        j = self.joints[name]
        return BLENDER_TO_MINECRAFT @ j.rest_local if j.is_root else j.rest_local

    def ef_to_origin(self, name) -> Mat4:
        """Joint.toOrigin: inverse of the Minecraft-space rest model matrix."""
        return self.rest_model_minecraft(name).inverse()

    def pose_model_minecraft(self, deltas) -> dict:
        """Model-space matrices Epic Fight would compute for a pose.

        deltas maps joint name -> the per-joint local delta (a TransformSheet value, and
        the same thing as Blender's pose_bone.matrix_basis). Missing joints get identity.
        Mirrors Armature.getPoseTransform / JointTransform.getAnimationBoundMatrix:
        M_j = M_parent @ localTransform_j @ delta_j.
        """
        out = {}

        def walk(name, parent):
            m = parent @ self.ef_local(name) @ deltas.get(name, Mat4.identity())
            out[name] = m
            for c in self.joints[name].children:
                walk(c, m)

        walk(self.root, Mat4.identity())
        return out

    def skinning_matrices(self, deltas) -> dict:
        """Model @ toOrigin - what actually multiplies a vertex."""
        model = self.pose_model_minecraft(deltas)
        return {n: model[n] @ self.ef_to_origin(n) for n in model}


    def groups(self) -> dict:
        return dict(BIPED_GROUPS) if set(self.names()) == set(_BIPED_NAMES) else {}


def _read_transform(transform, fmt) -> Mat4:
    """JsonAssetLoader sniffs the shape and overrides the declared format, so we do too."""
    if isinstance(transform, dict):
        return Mat4.compose(transform["loc"], transform["rot"], transform["sca"])
    if len(transform) != 16:
        raise ValueError("matrix transform must have 16 elements")
    return Mat4.from_flat(transform)


BIPED_GROUPS = {
    "root": ("Root",),
    "spine": ("Torso", "Chest", "Head"),
    "arm_r": ("Shoulder_R", "Arm_R", "Hand_R"),
    "arm_l": ("Shoulder_L", "Arm_L", "Hand_L"),
    "leg_r": ("Thigh_R", "Leg_R"),
    "leg_l": ("Thigh_L", "Leg_L"),
    "poles": ("Knee_R", "Knee_L", "Elbow_R", "Elbow_L"),
    "sockets": ("Tool_R", "Tool_L"),
}

_BIPED_NAMES = [n for g in BIPED_GROUPS.values() for n in g]
