"""SOMA, the body NVIDIA GEM-X solves for, and the motion it hands back. No bpy.

GEM-X (github.com/NVlabs/GEM-X) estimates a 77-joint SOMA body from a monocular video and
torch.saves the answer to <output>/<video>/preprocess/hpe_results.pt. The part an
animation needs is body_params_global:

  global_orient  (L, 3)    axis-angle, the Hips
  body_pose      (L, 228)  76 more axis-angles, SOMA joints 2..77 in order
  transl         (L, 3)    the Hips' world position, metres

in a gravity-aligned world, Y up, grounded by GEM's own post-process so the lowest joint
of the whole clip touches y = 0.

Every rotation is SMPL-style: relative to the T-pose, in a frame that is world-aligned
there. SOMA composes a joint as orient[parent]^-1 @ pose[j] @ orient[j] (soma.geometry.
rig_utils.apply_joint_orient_local), so the orients cancel down a chain and what is left
is one product,

    G[Hips] = R(global_orient)        G[j] = G[parent] @ R(pose[j])

G[j] is joint j's world rotation measured from its T-pose. A bone of the T-pose pointing
along d points along G @ d in the frame, which is all a retarget needs: no identity model,
no mesh, no torch. Positions follow from the T-pose offsets the same way.

A SOMA BVH (BONES-SEED, Kimodo, GEM-X's own export) carries W[j] = G[j] @ orient[j]
instead, so its G is W @ orient^-1 with the T-pose orients from data/soma77.json.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from . import quat as Q
from .tensorio import Opaque, Tensor, TensorIOError, read_npz, read_torch, walk

__all__ = ["SomaSkeleton", "SomaMotion", "SomaError", "load_motion", "motion_from_params",
           "motion_from_bvh", "find_body_params", "BODY_JOINTS", "SOMA_JOINT_COUNT",
           "DEFAULT_FPS"]

SOMA_JOINT_COUNT = 77

DEFAULT_FPS = 30.0

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "soma77.json")

BODY_JOINTS = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandMiddle1",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandMiddle1",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
)


class SomaError(ValueError):
    """A file that is not, or not quite, a SOMA motion. The message is for the user."""


class SomaSkeleton:
    """The 78-entry SOMA hierarchy (virtual Root plus 77 joints) in its T-pose."""

    _reference = None

    def __init__(self, names, parents, tpose_position, tpose_orient):
        self.names = list(names)
        self.parents = [int(p) for p in parents]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.tpose_position = [tuple(map(float, p)) for p in tpose_position]
        self.tpose_orient = [Q.normalize(tuple(map(float, q))) for q in tpose_orient]
        self.offsets = [
            p if self.parents[j] < 0 else Q.sub3(p, self.tpose_position[self.parents[j]])
            for j, p in enumerate(self.tpose_position)]

    @classmethod
    def reference(cls) -> "SomaSkeleton":
        if cls._reference is None:
            with open(TABLE, "r", encoding="utf-8") as fh:
                t = json.load(fh)
            cls._reference = cls(t["joints"], t["parents"], t["tpose_position"],
                                 t["tpose_orient"])
        return cls._reference

    def __len__(self):
        return len(self.names)

    def ancestors_closed(self, names) -> list:
        """Joint indices for `names` and every joint above them, parents first."""
        want = set()
        for n in names:
            j = self.index[n]
            while j >= 0 and j not in want:
                want.add(j)
                j = self.parents[j]
        return sorted(want)

    def direction(self, a, b):
        """Unit T-pose direction from joint a to joint b, SOMA world."""
        return Q.unit3(Q.sub3(self.tpose_position[self.index[b]],
                              self.tpose_position[self.index[a]]))


@dataclass
class SomaMotion:
    """GEM-X's answer reduced to what a retarget reads.

    rotations  per frame, 78 entries: G[j] as (w, x, y, z), None where not computed
    hips       per frame, the Hips' world position, metres
    offsets    78 T-pose offsets from the parent, metres, SOMA world (Y up, +Z forward)
    """
    fps: float
    rotations: list
    hips: list
    offsets: list
    skeleton: SomaSkeleton
    source: str = ""
    kind: str = ""
    notes: list = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.rotations)

    def joint(self, name) -> int:
        return self.skeleton.index[name]

    def positions(self, frame, names) -> dict:
        """World positions of `names` at `frame`, by forward kinematics from the Hips."""
        sk = self.skeleton
        rot = self.rotations[frame]
        hips = sk.index["Hips"]
        cache = {hips: tuple(self.hips[frame])}

        def pos(j):
            if j in cache:
                return cache[j]
            p = sk.parents[j]
            if p < 0:
                cache[j] = (0.0, 0.0, 0.0)
                return cache[j]
            out = Q.add3(pos(p), Q.rotate(rot[p], self.offsets[j]))
            cache[j] = out
            return out

        return {n: pos(sk.index[n]) for n in names}


def _squeeze(t: Tensor) -> Tensor:
    shape = list(t.shape)
    while len(shape) > 2 and shape[0] == 1:
        shape.pop(0)
    return t.reshape(shape) if shape != list(t.shape) else t


def _per_frame(t, name, widths):
    """(L, k) rows of a tensor whose frame is `widths` numbers wide, whatever the
    trailing axes are called: (L, 228) and (L, 76, 3) both come back as 228 a row."""
    t = _squeeze(t)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    frames = t.shape[0]
    width = t.numel // frames if frames else 0
    if width not in widths:
        raise SomaError("%s has shape %s; expected %s numbers per frame"
                        % (name, t.shape, " or ".join(str(w) for w in widths)))
    return t.reshape(frames, width).rows(width), width


def _rot_rows(row, width, count):
    """count rotations out of one frame's row: axis-angle (3 each) or 3x3 matrices."""
    if width == 3 * count:
        return [Q.from_rotvec(row[3 * i:3 * i + 3]) for i in range(count)]
    return [Q.from_rows((row[9 * i:9 * i + 3], row[9 * i + 3:9 * i + 6],
                         row[9 * i + 6:9 * i + 9])) for i in range(count)]


def motion_from_params(global_orient, body_pose, transl, fps=DEFAULT_FPS,
                       skeleton=None, joints=BODY_JOINTS, source="", kind="params"):
    """G and the Hips track from SOMA pose parameters. Tensors or nested lists.

    Only `joints` and the chain above them are composed; the fingers and the face are
    not, because the Epic Fight biped has nowhere to put them.
    """
    sk = skeleton or SomaSkeleton.reference()
    go = global_orient if isinstance(global_orient, Tensor) else _as_tensor(global_orient)
    bp = body_pose if isinstance(body_pose, Tensor) else _as_tensor(body_pose)
    tr = transl if isinstance(transl, Tensor) else _as_tensor(transl)
    go_rows, go_w = _per_frame(go, "global_orient", (3, 9))
    bp_rows, bp_w = _per_frame(bp, "body_pose", (3 * 76, 9 * 76))
    tr_rows, _ = _per_frame(tr, "transl", (3,))
    frames = min(len(go_rows), len(bp_rows), len(tr_rows))
    if frames == 0:
        raise SomaError("the pose parameters hold no frames")
    notes = []
    if len({len(go_rows), len(bp_rows), len(tr_rows)}) != 1:
        notes.append("parameter lengths differ (%d/%d/%d); using the first %d frames"
                     % (len(go_rows), len(bp_rows), len(tr_rows), frames))

    wanted = sk.ancestors_closed(joints)
    hips = sk.index["Hips"]
    rotations, positions = [], []
    for f in range(frames):
        local = _rot_rows(bp_rows[f], bp_w, 76)
        g = [None] * len(sk)
        g[0] = Q.IDENTITY
        g[hips] = _rot_rows(go_rows[f], go_w, 1)[0]
        for j in wanted:
            if j <= hips:
                continue
            g[j] = Q.normalize(Q.mul(g[sk.parents[j]], local[j - 2]))
        rotations.append(g)
        positions.append(tuple(float(v) for v in tr_rows[f]))

    scale = _unit_scale(positions)
    if scale != 1.0:
        positions = [Q.scale3(p, scale) for p in positions]
        notes.append("transl read as centimetres")
    return SomaMotion(float(fps), rotations, positions, list(sk.offsets), sk,
                      source=source, kind=kind, notes=notes)


def _unit_scale(positions) -> float:
    """1 for metres, 0.01 for a Hips track that is plainly centimetres."""
    heights = sorted(abs(p[1]) for p in positions)
    return 0.01 if heights and heights[len(heights) // 2] > 10.0 else 1.0


def _as_tensor(v) -> Tensor:
    flat, shape = [], []

    def dims(x):
        while isinstance(x, (list, tuple)):
            shape.append(len(x))
            x = x[0] if x else None

    def fill(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                fill(y)
        else:
            flat.append(float(x))

    dims(v)
    fill(v)
    return Tensor(shape, flat)


_REQUIRED = ("global_orient", "body_pose", "transl")


def _mapping(obj):
    if isinstance(obj, Opaque):
        return obj.fields()
    return obj if isinstance(obj, dict) else None


def find_body_params(obj, which="global"):
    """(params dict, dotted path) out of whatever GEM-X saved.

    hpe_results.pt is {"body_params_global": {...}, "body_params_incam": {...},
    "K_fullimg": ..., "net_outputs": {...}}. A dict that already is the params, an npz
    with the three keys at top level, and a "poses" (L, 77, 3) array are all accepted.
    """
    top = _mapping(obj)
    if top is None:
        raise SomaError("the file holds a %s, not a dict of GEM-X results"
                        % type(obj).__name__)
    want = "body_params_" + which
    candidates = [want, "pred_" + want, "smpl_params_" + which]
    for key in candidates:
        got = _mapping(top.get(key))
        if got is not None and all(k in got for k in _REQUIRED):
            return got, key
    for key, val in top.items():
        got = _mapping(val)
        if str(key).endswith("_params_" + which) and got is not None \
                and all(k in got for k in _REQUIRED):
            return got, str(key)
    if all(k in top for k in _REQUIRED):
        return top, ""
    if "poses" in top and "transl" in top:
        poses = _squeeze(top["poses"])
        frames = poses.shape[0]
        width = poses.numel // frames
        per = width // SOMA_JOINT_COUNT
        rows = poses.reshape(frames, width).rows(width)
        return {"global_orient": Tensor((frames, per), [v for r in rows for v in r[:per]]),
                "body_pose": Tensor((frames, width - per),
                                    [v for r in rows for v in r[per:]]),
                "transl": top["transl"]}, "poses"
    for key, val in top.items():
        got = _mapping(val)
        if got is None:
            continue
        try:
            params, path = find_body_params(got, which)
        except SomaError:
            continue
        return params, ("%s.%s" % (key, path)).strip(".")
    shown = ", ".join(sorted(str(k) for k in top)[:12])
    raise SomaError("no GEM-X body parameters (global_orient, body_pose, transl) in this "
                    "file; its top level holds: %s" % (shown or "nothing"))


def _load_tree(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npz":
        return read_npz(path)
    try:
        return read_torch(path)
    except TensorIOError as exc:
        raise SomaError(str(exc)) from None


def motion_from_file(path, fps=None, which="global"):
    tree = _load_tree(path)
    params, where = find_body_params(tree, which)
    rate = fps
    if not rate:
        for key in ("fps", "mocap_framerate", "framerate"):
            v = _mapping(tree).get(key) if _mapping(tree) else None
            if isinstance(v, Tensor) and v.numel == 1:
                rate = float(v.item())
                break
            if isinstance(v, (int, float)):
                rate = float(v)
                break
    motion = motion_from_params(params["global_orient"], params["body_pose"],
                                params["transl"], rate or DEFAULT_FPS,
                                source=str(path), kind="gem-x")
    if where:
        motion.notes.insert(0, "read %s" % where)
    if not rate:
        motion.notes.append("no frame rate in the file; assumed %g fps" % DEFAULT_FPS)
    return motion


def _euler_quat(channels, values):
    q = Q.IDENTITY
    for ch, v in zip(channels, values):
        axis = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[ch[0]]
        q = Q.mul(q, Q.from_axis_angle(axis, math.radians(v)))
    return q


def _parse_bvh(text):
    toks = text.split()
    names, parents, offsets, channels = [], [], [], []
    stack, i = [], 0
    n = len(toks)
    while i < n and toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            names.append(toks[i + 1].split(":")[-1].split("|")[-1])
            parents.append(stack[-1] if stack else -1)
            offsets.append((0.0, 0.0, 0.0))
            channels.append([])
            i += 2
        elif t == "End":
            depth, i = 0, i + 2
            while i < n:
                depth += toks[i] == "{"
                depth -= toks[i] == "}"
                i += 1
                if depth == 0:
                    break
        elif t == "{":
            stack.append(len(names) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            offsets[-1] = tuple(float(v) for v in toks[i + 1:i + 4])
            i += 4
            while i < n and _is_number(toks[i]):
                i += 1
        elif t == "CHANNELS":
            count = int(toks[i + 1])
            channels[-1] = toks[i + 2:i + 2 + count]
            i += 2 + count
        else:
            i += 1
    if i >= n:
        raise SomaError("not a BVH file: no MOTION section")
    frames = int(toks[i + 2])
    frame_time = float(toks[i + 5])
    width = sum(len(c) for c in channels)
    data = toks[i + 6:]
    if len(data) < frames * width:
        frames = len(data) // width if width else 0
    rows = [[float(v) for v in data[f * width:(f + 1) * width]] for f in range(frames)]
    return names, parents, offsets, channels, rows, frame_time


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def motion_from_bvh(path, fps=None, skeleton=None):
    """A SOMA-skeleton BVH (BONES-SEED, Kimodo, GEM-X / soma-retargeter export).

    The file's own offsets are used for positions, so its proportions survive; its
    rotations are turned into G with the T-pose orients, which is the convention those
    exporters write (local = orient[parent]^-1 @ G-step @ orient[j]).
    """
    sk = skeleton or SomaSkeleton.reference()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        names, parents, offsets, channels, rows, frame_time = _parse_bvh(fh.read())
    missing = [n for n in BODY_JOINTS if n not in names]
    if missing:
        raise SomaError("this BVH is not a SOMA skeleton: no %s joint%s"
                        % (", ".join(missing[:4]), "s" if len(missing) > 1 else ""))
    if not rows:
        raise SomaError("the BVH has no frames")

    local_pos = list(offsets)

    # BVH index -> SOMA index; the BVH may or may not carry the virtual Root.
    to_soma = [sk.index.get(n, -1) for n in names]
    orient = sk.tpose_orient
    hips_b = names.index("Hips")

    rotations, hips = [], []
    for row in rows:
        world_q = [None] * len(names)
        world_p = [None] * len(names)
        c = 0
        for b, chans in enumerate(channels):
            vals = row[c:c + len(chans)]
            c += len(chans)
            pos = list(local_pos[b])
            rot_ch, rot_v = [], []
            for ch, v in zip(chans, vals):
                if ch.endswith("position"):
                    pos["XYZ".index(ch[0])] = v
                elif ch.endswith("rotation"):
                    rot_ch.append(ch)
                    rot_v.append(v)
            q = _euler_quat(rot_ch, rot_v)
            p = parents[b]
            if p < 0:
                world_q[b], world_p[b] = q, pos
            else:
                world_q[b] = Q.mul(world_q[p], q)
                world_p[b] = Q.add3(world_p[p], Q.rotate(world_q[p], pos))
        g = [None] * len(sk)
        g[0] = Q.IDENTITY
        for b, j in enumerate(to_soma):
            if j >= 0:
                g[j] = Q.normalize(Q.mul(world_q[b], Q.conj(orient[j])))
        rotations.append(g)
        hips.append(world_p[hips_b])

    scale = _unit_scale(hips)
    hips = [Q.scale3(p, scale) for p in hips]

    # T-pose world offsets from the file's own bone lengths.
    file_offsets = list(sk.offsets)
    for b, j in enumerate(to_soma):
        p = parents[b]
        if j < 0 or p < 0 or to_soma[p] < 0:
            continue
        file_offsets[j] = Q.rotate(orient[to_soma[p]], Q.scale3(local_pos[b], scale))

    rate = fps or (1.0 / frame_time if frame_time > 0 else DEFAULT_FPS)
    motion = SomaMotion(float(rate), rotations, hips, file_offsets, sk,
                        source=str(path), kind="bvh")
    if scale != 1.0:
        motion.notes.append("BVH read as centimetres")
    return motion


def load_motion(path, fps=None, which="global") -> SomaMotion:
    """Any SOMA motion this add-on understands: GEM-X hpe_results.pt (or any torch file
    or npz holding global_orient / body_pose / transl), or a SOMA-skeleton BVH."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".bvh":
        return motion_from_bvh(path, fps)
    return motion_from_file(path, fps, which)


def describe(path) -> str:
    """One line per leaf of a file's tree, for the error a wrong file deserves."""
    tree = _load_tree(path)
    out = []
    for name, value in walk(tree):
        shape = getattr(value, "shape", None)
        out.append("%s: %s" % (name, "tensor %s" % (shape,) if shape is not None
                               else type(value).__name__))
    return "\n".join(out)
