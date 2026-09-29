"""The `vertices` block of an Epic Fight animmodel json - biped.json and its family.

The file stores Blender coordinates. JsonAssetLoader.loadSkinnedMesh multiplies every
position and normal by BLENDER_TO_MINECRAFT at load time, so the raw arrays go straight
into Blender with no basis change and land on the armature's rest pose.

Layout:
  positions  stride 3, one per vertex
  normals    stride 3, an independent pool indexed per corner
  uvs        stride 2, likewise, and V runs from the TOP of the sheet
  vcounts    stride 1, per vertex: how many joints affect it
  vindices   stride 1, flat (jointId, weightIndex) pairs grouped by vcounts
  weights    stride 1, the weight value pool
  parts      name -> flat (position, uv, normal) triples, three per triangle
  indices    the same thing as a single unnamed part, in older files

No numpy: efb stays importable under a bare python3 so the unit gate needs no Blender.
"""

from __future__ import annotations

import json

__all__ = ["BodyMesh", "MeshPart", "load_mesh", "BIPED_JOINTS", "UNGROUPED"]

BIPED_JOINTS = ["Root", "Thigh_R", "Leg_R", "Knee_R", "Thigh_L", "Leg_L", "Knee_L",
                "Torso", "Chest", "Head", "Shoulder_R", "Arm_R", "Hand_R", "Tool_R",
                "Elbow_R", "Shoulder_L", "Arm_L", "Hand_L", "Tool_L", "Elbow_L"]

UNGROUPED = "noGroups"
COORD_BONE = "Coord"


class MeshPart:
    """One named group of triangles. `corners` is a list of (pos, uv, normal) triples."""

    __slots__ = ("name", "corners")

    def __init__(self, name, corners):
        self.name = name
        self.corners = corners

    @property
    def triangles(self) -> int:
        return len(self.corners) // 3


class BodyMesh:
    def __init__(self, name, positions, normals, uvs, parts, skin, joints):
        self.name = name
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.parts = parts
        self.skin = skin
        self.joints = joints

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return sum(p.triangles for p in self.parts)

    def bounds(self):
        lo = [min(p[i] for p in self.positions) for i in range(3)]
        hi = [max(p[i] for p in self.positions) for i in range(3)]
        return lo, hi

    def used_joints(self):
        out = []
        for infl in self.skin:
            for name, _w in infl:
                if name not in out:
                    out.append(name)
        return out

    def loops(self):
        """Flatten every part into what a Blender mesh consumes directly.

        -> (faces, loop_uvs, loop_normals, face_parts), one entry per triangle corner in
        face order. Faces index `positions`; uvs and normals are per corner because the
        json pools them separately.
        """
        faces, loop_uv, loop_n, owner = [], [], [], []
        for part in self.parts:
            for t in range(part.triangles):
                tri = part.corners[t * 3:t * 3 + 3]
                faces.append(tuple(c[0] for c in tri))
                for _p, u, n in tri:
                    loop_uv.append(self.uvs[u])
                    loop_n.append(self.normals[n])
                owner.append(part.name)
        return faces, loop_uv, loop_n, owner

    def group_weights(self):
        """joint name -> list of (vertex index, weight)."""
        out = {}
        for vi, infl in enumerate(self.skin):
            for jname, weight in infl:
                out.setdefault(jname, []).append((vi, weight))
        return out


def _chunk(node, stride):
    flat = node["array"]
    if stride == 1:
        return list(flat)
    return [tuple(flat[i:i + stride]) for i in range(0, len(flat), stride)]


def load_mesh(data, name="mesh", joint_names=None) -> BodyMesh:
    """`data` is bytes, str or an already parsed dict."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8-sig")
    if isinstance(data, str):
        data = json.loads(data)
    if "parent" in data:
        raise ValueError("%s inherits from %s; resolve the parent first"
                         % (name, data["parent"]))
    if "vertices" not in data:
        raise ValueError("%s carries no vertices block" % name)

    v = data["vertices"]
    positions = [tuple(float(c) for c in p) for p in _chunk(v["positions"], 3)]
    normals = [tuple(float(c) for c in p) for p in _chunk(v["normals"], 3)]
    uvs = [tuple(float(c) for c in p) for p in _chunk(v["uvs"], 2)]

    if joint_names is None:
        arm = data.get("armature")
        joint_names = ([n for n in arm["joints"] if n != COORD_BONE] if arm
                       else list(BIPED_JOINTS))

    parts = []
    for pname, pnode in (v.get("parts") or {}).items():
        parts.append(MeshPart(pname, _chunk(pnode, 3)))
    if "indices" in v:
        parts.append(MeshPart(UNGROUPED, _chunk(v["indices"], 3)))
    parts.sort(key=lambda p: p.name)

    skin = _read_skin(v, joint_names, name) if "vcounts" in v else [[] for _ in positions]
    if len(skin) != len(positions):
        raise ValueError("%s: %d skin entries for %d positions"
                         % (name, len(skin), len(positions)))
    return BodyMesh(name, positions, normals, uvs, parts, skin, joint_names)


def _read_skin(v, joint_names, name):
    vcounts = _chunk(v["vcounts"], 1)
    vindices = _chunk(v["vindices"], 1)
    weights = _chunk(v["weights"], 1)
    skin, cursor = [], 0
    for count in vcounts:
        infl = []
        for _ in range(int(count)):
            jid = int(vindices[cursor * 2])
            wid = int(vindices[cursor * 2 + 1])
            cursor += 1
            jname = joint_names[jid] if jid < len(joint_names) else "joint_%d" % jid
            infl.append((jname, float(weights[wid])))
        skin.append(infl)
    if cursor * 2 != len(vindices):
        raise ValueError("%s: vindices length %d does not match vcounts sum %d"
                         % (name, len(vindices), cursor * 2))
    return skin
