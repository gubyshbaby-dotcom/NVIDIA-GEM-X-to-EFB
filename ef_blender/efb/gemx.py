"""NVIDIA GEM-X (SOMA) motion -> an Epic Fight animation document for the biped. No bpy.

The translator. It reads what efb.soma hands it - per frame, each SOMA joint's world
rotation measured from the T-pose, and the Hips' track - and writes the same thing the
game reads: an ATTRIBUTES-format document, one local delta per joint per frame. From there
it is an ordinary Epic Fight clip: efbpy.animimport keys it onto the control rig exactly as
it would a shipped animation, IK bake included, and efb.animjson can write it straight out
for the mod.

How a pose crosses over
-----------------------
Both skeletons are brought into the rig's armature space (Blender, Z up, the biped facing
+Y, its right hand on +X). SOMA's world is Y up, facing +Z, left on +X; the map between
them is the proper rotation SOMA_TO_EF.

Each Epic Fight bone copies one SOMA joint's world rotation-from-rest:

    pose[bone] = G[joint] @ align[bone] @ rest[bone]

`align` is what makes two different rest poses agree. SOMA's rests in a T, the biped with
its arms hanging, so an arm or a leg bone is first swung, by the shortest rotation, onto the
direction its SOMA segment has in the T-pose; the spine, the head and the collarbones rest
alike and take the identity. At the T-pose G is the identity, so the biped stands in the
T-pose too, and every motion is then carried over as a turn from there.

The elbow and knee are hinges on the biped - one local X axis, a stop on either side, a pin
on the other two - and the rig keeps them that way. A human forearm is not: it pronates.
So by default the limb is solved rather than copied: the upper bone keeps its SOMA
direction and is turned about it until its hinge axis stands normal to the plane of the
limb, and the lower bone then folds about that axis alone onto its SOMA direction. The
wrist's real orientation, pronation included, goes to the Tool socket instead, which is
where the weapon sits. Near a straight limb the plane is noise, so the correction fades out
below HINGE_FADE degrees of bend and the copied twist stands.

The Hips' track is scaled by leg length - the biped's legs are 0.761 m, a person's about
0.93 - so stride and jump height fit the body that has to perform them, and the result is
dropped onto the floor through the biped's own feet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from . import quat as Q
from .animjson import ATTRIBUTES, AnimationDocument, TRS, Track
from .armature import Armature
from .clip import EF_TIME_DECIMALS, EF_VALUE_DECIMALS, quantize_trs, time_from_frame
from .rigdef import LENGTHS
from .rigedit import SEAMS, seam_slide
from .soma import SomaError, SomaMotion, SomaSkeleton, load_motion

__all__ = ["RetargetOptions", "RetargetReport", "retarget", "convert_file",
           "SOMA_TO_EF", "BODY_SOURCES", "LIMBS", "ROOT_FULL", "ROOT_IN_PLACE",
           "HINGE_FADE", "required_joints", "check_armature", "SomaError"]

SOMA_TO_EF = Q.from_rows(((-1.0, 0.0, 0.0),
                          (0.0, 0.0, 1.0),
                          (0.0, 1.0, 0.0)))

ROOT_FULL = "FULL"
ROOT_IN_PLACE = "IN_PLACE"

HINGE_FADE = (8.0, 20.0)

FORWARD = (0.0, 1.0, 0.0)

MAX_AUTO_FPS = 60.0

BODY_SOURCES = {
    "Root": ("Hips",),
    "Torso": ("Spine1", "Spine2"),
    "Chest": ("Chest",),
    "Head": ("Head",),
    "Shoulder_R": ("RightShoulder",),
    "Shoulder_L": ("LeftShoulder",),
}


@dataclass(frozen=True)
class Limb:
    upper: str
    lower: str
    tool: str
    upper_src: tuple
    lower_src: tuple
    wrist: str
    fold: float


LIMBS = (
    Limb("Arm_R", "Hand_R", "Tool_R", ("RightArm", "RightForeArm"),
         ("RightForeArm", "RightHand"), "RightHand", 1.0),
    Limb("Arm_L", "Hand_L", "Tool_L", ("LeftArm", "LeftForeArm"),
         ("LeftForeArm", "LeftHand"), "LeftHand", 1.0),
    Limb("Thigh_R", "Leg_R", "", ("RightLeg", "RightShin"),
         ("RightShin", "RightFoot"), "", -1.0),
    Limb("Thigh_L", "Leg_L", "", ("LeftLeg", "LeftShin"),
         ("LeftShin", "LeftFoot"), "", -1.0),
)

FEET = ("Leg_R", "Leg_L")

SOMA_FEET = ("LeftFoot", "LeftToeBase", "LeftToeEnd",
             "RightFoot", "RightToeBase", "RightToeEnd")
SOMA_HIPS = ("LeftLeg", "RightLeg")


@dataclass
class RetargetOptions:
    """Everything the operator and the command line let a user change.

    fps             output rate; 0 keeps the source's, capped at MAX_AUTO_FPS
    start, end      source frames to use, inclusive; end -1 is the last
    smoothing       gaussian sigma in source frames, 0 off. GEM-X regresses a whole clip
                    at once and is steady; a little takes the last shimmer off hands
    root_motion     FULL keeps the path, IN_PLACE keeps only the height
    face_forward    turn the clip so its first frame faces the rig's front
    start_at_origin slide it so its first frame stands on the rig's origin
    ground          drop it so the biped's feet touch z = 0
    subject_height  the filmed person's height in metres, 0 estimates it
    stride          horizontal scale, 0 is the leg-length ratio
    hinge           solve elbows and knees as hinges (see the module docstring)
    tools           give the wrist's orientation to Tool_R / Tool_L
    clavicle        how much of the collarbones' motion the shoulders take, 0..1
    markers         write Elbow_* / Knee_* seam tracks - for a json going straight into
                    the game; the Blender rig slides them itself
    static_tracks   write rest tracks for joints nothing drives, as shipped clips do
    """
    fps: float = 0.0
    start: int = 0
    end: int = -1
    smoothing: float = 1.0
    root_motion: str = ROOT_FULL
    face_forward: bool = True
    start_at_origin: bool = True
    ground: bool = True
    subject_height: float = 0.0
    stride: float = 0.0
    hinge: bool = True
    tools: bool = True
    clavicle: float = 1.0
    markers: bool = False
    static_tracks: bool = False


@dataclass
class RetargetReport:
    source: str = ""
    kind: str = ""
    frames_in: int = 0
    frames_out: int = 0
    fps_in: float = 0.0
    fps_out: float = 0.0
    subject_height: float = 0.0
    height_estimated: bool = False
    leg_scale: float = 1.0
    floor_offset: float = 0.0
    turned_degrees: float = 0.0
    hinge_error_degrees: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (self.frames_out - 1) / self.fps_out if self.fps_out and self.frames_out else 0.0

    def summary(self) -> str:
        return ("%d frames at %g fps (%.2f s), subject %.2f m%s, legs x%.3f"
                % (self.frames_out, self.fps_out, self.duration, self.subject_height,
                   " (estimated)" if self.height_estimated else "", self.leg_scale))


def required_joints():
    out = set(BODY_SOURCES)
    for limb in LIMBS:
        out.update(n for n in (limb.upper, limb.lower, limb.tool) if n)
    return out


def check_armature(armature: Armature):
    missing = sorted(n for n in required_joints() if n not in armature)
    if missing:
        raise SomaError("%s is not the Epic Fight biped: it has no %s"
                        % (armature.name, ", ".join(missing[:5])))


def _to_ef(q):
    return Q.mul(Q.mul(SOMA_TO_EF, q), Q.conj(SOMA_TO_EF))


def _needed(sk: SomaSkeleton):
    names = set(SOMA_FEET) | set(SOMA_HIPS) | {"Hips"}
    for src in BODY_SOURCES.values():
        names.update(src)
    for limb in LIMBS:
        names.update(limb.upper_src + limb.lower_src)
        if limb.wrist:
            names.add(limb.wrist)
    return sk.ancestors_closed(sorted(names))


# -- time: window, smoothing, resampling -------------------------------------------------

def _window(motion: SomaMotion, start, end) -> SomaMotion:
    n = motion.frame_count
    lo = max(0, min(int(start), n - 1))
    hi = n - 1 if end is None or end < 0 else max(lo, min(int(end), n - 1))
    if lo == 0 and hi == n - 1:
        return motion
    return replace(motion, rotations=motion.rotations[lo:hi + 1],
                   hips=motion.hips[lo:hi + 1])


def _gaussian(sigma):
    radius = max(1, int(math.ceil(2.5 * sigma)))
    w = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    total = sum(w)
    return radius, [v / total for v in w]


def _smooth(motion: SomaMotion, sigma, joints) -> SomaMotion:
    if sigma <= 0.0 or motion.frame_count < 3:
        return motion
    radius, weights = _gaussian(sigma)
    n = motion.frame_count
    rots = [list(r) for r in motion.rotations]
    for j in joints:
        track = [motion.rotations[f][j] for f in range(n)]
        if any(q is None for q in track):
            continue
        for f in range(n):
            centre = track[f]
            acc = [0.0, 0.0, 0.0, 0.0]
            for k, w in zip(range(-radius, radius + 1), weights):
                q = Q.align(track[min(n - 1, max(0, f + k))], centre)
                for i in range(4):
                    acc[i] += w * q[i]
            rots[f][j] = Q.normalize(tuple(acc))
    hips = []
    for f in range(n):
        acc = [0.0, 0.0, 0.0]
        for k, w in zip(range(-radius, radius + 1), weights):
            p = motion.hips[min(n - 1, max(0, f + k))]
            for i in range(3):
                acc[i] += w * p[i]
        hips.append(tuple(acc))
    return replace(motion, rotations=rots, hips=hips)


def _resample(motion: SomaMotion, fps_out, joints) -> SomaMotion:
    if abs(fps_out - motion.fps) < 1e-6 or motion.frame_count < 2:
        return replace(motion, fps=fps_out)
    n = motion.frame_count
    count = int(math.floor((n - 1) * fps_out / motion.fps + 1e-9)) + 1
    rots, hips = [], []
    for k in range(count):
        t = k * motion.fps / fps_out
        i = min(int(math.floor(t)), n - 2)
        u = t - i
        a, b = motion.rotations[i], motion.rotations[i + 1]
        row = [None] * len(a)
        row[0] = Q.IDENTITY
        for j in joints:
            if a[j] is not None and b[j] is not None:
                row[j] = Q.slerp(a[j], b[j], u)
        rots.append(row)
        hips.append(Q.lerp3(motion.hips[i], motion.hips[i + 1], u))
    return replace(motion, rotations=rots, hips=hips, fps=fps_out)


def _upright(motion: SomaMotion):
    """(motion, flipped). A clip whose heads hang below its hips is in camera space - Y
    down, what GEM-X's ONNX demo falls back to when it has no world solve - and is turned
    half a turn about X so it stands."""
    step = max(1, motion.frame_count // 50)
    lift = 0.0
    for f in range(0, motion.frame_count, step):
        p = motion.positions(f, ("Hips", "Head"))
        lift += p["Head"][1] - p["Hips"][1]
    if lift >= 0.0:
        return motion, False
    flip = Q.from_axis_angle((1.0, 0.0, 0.0), math.pi)
    rots = [[None if q is None else (q if j == 0 else Q.mul(flip, q))
             for j, q in enumerate(row)] for row in motion.rotations]
    hips = [Q.rotate(flip, p) for p in motion.hips]
    return replace(motion, rotations=rots, hips=hips), True


def _pick_fps(requested, source):
    if requested and requested > 0.0:
        return float(requested)
    if source <= MAX_AUTO_FPS + 1e-6:
        return float(round(source)) if abs(source - round(source)) < 0.05 else source
    for cand in (60.0, 30.0):
        if abs(source / cand - round(source / cand)) < 1e-3:
            return cand
    return MAX_AUTO_FPS


# -- the body: sizes -------------------------------------------------------------------

def _tpose_positions(motion: SomaMotion):
    sk = motion.skeleton
    pos = {}
    for j in range(len(sk)):
        p = sk.parents[j]
        pos[j] = motion.offsets[j] if p < 0 else Q.add3(pos[p], motion.offsets[j])
    return pos


def _leg_length(motion: SomaMotion):
    """Hip joint to sole, T-pose, in the motion's own offsets."""
    sk = motion.skeleton
    pos = _tpose_positions(motion)
    out = []
    for side in ("Left", "Right"):
        hip = pos[sk.index[side + "Leg"]][1]
        sole = min(pos[sk.index[side + n]][1] for n in ("Foot", "ToeBase", "ToeEnd"))
        out.append(hip - sole)
    return sum(out) / len(out)


def _stature(motion: SomaMotion):
    """Sole to top of head, T-pose."""
    sk = motion.skeleton
    pos = _tpose_positions(motion)
    sole = min(pos[sk.index[n]][1] for n in SOMA_FEET)
    return pos[sk.index["HeadEnd"]][1] - sole


def _quantile(values, q):
    got = sorted(values)
    if not got:
        return 0.0
    k = min(len(got) - 1, max(0, int(round(q * (len(got) - 1)))))
    return got[k]


GROUNDED = 0.15


def _estimate_size(motion: SomaMotion, frames_pos):
    """How much taller the filmed person is than the reference skeleton, or None.

    GEM-X's post-process grounds a clip so its lowest joint touches y = 0 at least once,
    measured on the person's real skeleton. On the reference skeleton the same pose puts
    the feet `drop` below the hips, so hips_height / drop is the size ratio on every frame
    a foot is down and larger on every other; a low quantile finds the ones that are.

    Only a grounded clip can answer. The ONNX demo skips that post-process, so when the
    reference feet do not come within GROUNDED metres of y = 0 there is no floor to
    measure against and the answer is None.
    """
    lows = [min(feet) for _hips, feet in frames_pos]
    if abs(_quantile(lows, 0.02)) > GROUNDED:
        return None
    ratios = []
    for hips_y, feet in frames_pos:
        drop = hips_y - min(feet)
        if drop > 0.3 and hips_y > 0.0:
            ratios.append(hips_y / drop)
    if len(ratios) < 3:
        return None
    k = _quantile(ratios, 0.05)
    return k if 0.75 <= k <= 1.3 else None


# -- the per-frame solve ----------------------------------------------------------------

@dataclass
class _Rest:
    rot: dict
    head: dict
    local_rot: dict
    local_t: dict
    parent: dict
    order: list


def _rest(armature: Armature) -> _Rest:
    r = _Rest({}, {}, {}, {}, {}, armature.depth_first())
    for name in r.order:
        m = armature.rest_model(name)
        r.rot[name] = m.to_quaternion()
        r.head[name] = m.to_translation()
        j = armature[name]
        r.parent[name] = j.parent
        r.local_rot[name] = j.rest_local.to_quaternion()
        r.local_t[name] = j.rest_local.to_translation()
    return r


def _fade(bend_degrees):
    lo, hi = HINGE_FADE
    t = min(1.0, max(0.0, (bend_degrees - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def _x_angle(q):
    """Local X of a rotation as Blender's XYZ euler reads it, degrees."""
    w, x, y, z = q
    return math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


class _Solver:
    def __init__(self, motion: SomaMotion, armature: Armature, opts: RetargetOptions):
        self.m = motion
        self.sk = motion.skeleton
        self.rest = _rest(armature)
        self.opts = opts
        ix = self.sk.index
        self.ix = ix
        self.align = {}
        self.src_dir = {}
        for limb in LIMBS:
            for bone, (a, b) in ((limb.upper, limb.upper_src), (limb.lower, limb.lower_src)):
                d = Q.rotate(SOMA_TO_EF, self.sk.direction(a, b))
                self.src_dir[bone] = d
                self.align[bone] = Q.between(self._bone_y(bone), d)
        self.rest_fold = {}
        for limb in LIMBS:
            ql = self.rest.local_rot[limb.lower]
            self.rest_fold[limb.lower] = Q.signed_angle(
                (0.0, 1.0, 0.0), Q.rotate(ql, (0.0, 1.0, 0.0)), (1.0, 0.0, 0.0))
        self.hinge_error = 0.0

    def _bone_y(self, bone):
        return Q.rotate(self.rest.rot[bone], (0.0, 1.0, 0.0))

    def g(self, frame, name, yaw):
        return Q.mul(yaw, _to_ef(self.m.rotations[frame][self.ix[name]]))

    def pose(self, frame, yaw):
        """World rotations of every driven biped bone at one frame, armature space."""
        rest, opts = self.rest, self.opts
        out = {}
        for bone, src in BODY_SOURCES.items():
            gs = [self.g(frame, s, yaw) for s in src]
            g = gs[0] if len(gs) == 1 else Q.nlerp(gs[0], gs[1], 0.5)
            if bone.startswith("Shoulder") and opts.clavicle < 1.0:
                chest = self.g(frame, "Chest", yaw)
                g = Q.nlerp(chest, g, max(0.0, opts.clavicle))
            out[bone] = Q.normalize(Q.mul(g, rest.rot[bone]))
        for limb in LIMBS:
            g_up = self.g(frame, limb.upper_src[0], yaw)
            g_low = self.g(frame, limb.lower_src[0], yaw)
            up = Q.normalize(Q.mul(g_up, Q.mul(self.align[limb.upper], rest.rot[limb.upper])))
            low = Q.normalize(Q.mul(g_low, Q.mul(self.align[limb.lower],
                                                 rest.rot[limb.lower])))
            if opts.hinge:
                up, low = self._hinge(limb, up, Q.rotate(g_low, self.src_dir[limb.lower]))
            out[limb.upper], out[limb.lower] = up, low
            if limb.tool and limb.tool in rest.rot:
                if opts.tools:
                    g_wr = self.g(frame, limb.wrist, yaw)
                    out[limb.tool] = Q.normalize(
                        Q.mul(g_wr, Q.mul(self.align[limb.lower], rest.rot[limb.tool])))
                else:
                    out[limb.tool] = Q.normalize(
                        Q.mul(low, Q.mul(Q.conj(rest.rot[limb.lower]), rest.rot[limb.tool])))
        return out

    def _hinge(self, limb, up, want_low_dir):
        """Turn the upper bone about itself until the limb's bend lies in its hinge plane,
        then fold the lower bone about the hinge alone."""
        rest = self.rest
        u = Q.rotate(up, (0.0, 1.0, 0.0))
        l = Q.unit3(want_low_dir)
        n = Q.cross3(u, l)
        bend = math.degrees(math.atan2(Q.norm3(n), Q.dot3(u, l)))
        w = _fade(bend)
        if w > 0.0:
            x_now = Q.rotate(up, (1.0, 0.0, 0.0))
            x_want = Q.scale3(Q.unit3(n), limb.fold)
            tau = Q.signed_angle(x_now, x_want, u)
            up = Q.normalize(Q.mul(Q.from_axis_angle(u, w * tau), up))
        x_axis = Q.rotate(up, (1.0, 0.0, 0.0))
        phi = Q.signed_angle(u, l, x_axis)
        fold = phi - self.rest_fold[limb.lower]
        local = Q.mul(rest.local_rot[limb.lower], Q.from_axis_angle((1.0, 0.0, 0.0), fold))
        low = Q.normalize(Q.mul(up, local))
        got = Q.rotate(low, (0.0, 1.0, 0.0))
        err = math.degrees(math.acos(max(-1.0, min(1.0, Q.dot3(got, l)))))
        self.hinge_error = max(self.hinge_error, err)
        return up, low

    def heads(self, world, root_pos):
        """Armature-space heads of the biped's bones for a pose, from the root down."""
        rest = self.rest
        heads = {}
        for name in rest.order:
            p = rest.parent[name]
            if p is None:
                heads[name] = root_pos
                continue
            prot = world.get(p) or self._rest_world(world, p)
            heads[name] = Q.add3(heads[p], Q.rotate(prot, rest.local_t[name]))
        return heads

    def _rest_world(self, world, name):
        """World rotation of an undriven bone: its parent's, carried by its rest local."""
        chain = []
        cur = name
        while cur is not None and cur not in world:
            chain.append(cur)
            cur = self.rest.parent[cur]
        q = world[cur] if cur is not None else Q.IDENTITY
        for n in reversed(chain):
            q = Q.mul(q, self.rest.local_rot[n])
            world[n] = q
        return world[name]

    def foot_heights(self, world, root_pos):
        world = dict(world)
        heads = self.heads(world, root_pos)
        out = []
        for leg in FEET:
            if leg in heads:
                tip = Q.add3(heads[leg], Q.rotate(world[leg], (0.0, LENGTHS.get(leg, 0.3864),
                                                               0.0)))
                out.append(min(tip[2], heads[leg][2]))
        return out


def retarget(motion: SomaMotion, armature: Armature, options=None):
    """(AnimationDocument, RetargetReport) for `motion` on `armature`."""
    opts = options or RetargetOptions()
    check_armature(armature)
    report = RetargetReport(source=motion.source, kind=motion.kind,
                            frames_in=motion.frame_count, fps_in=motion.fps,
                            notes=list(motion.notes))
    joints = _needed(motion.skeleton)
    motion, flipped = _upright(motion)
    if flipped:
        report.notes.append("the clip was upside down (camera space, Y down); stood it up")
    motion = _window(motion, opts.start, opts.end)
    motion = _smooth(motion, float(opts.smoothing or 0.0), joints)
    motion = _resample(motion, _pick_fps(opts.fps, motion.fps), joints)
    n = motion.frame_count
    if n == 0:
        raise SomaError("no frames left to convert")
    report.frames_out, report.fps_out = n, motion.fps

    solver = _Solver(motion, armature, opts)
    ix = motion.skeleton.index

    yaw = Q.IDENTITY
    if opts.face_forward:
        f = Q.rotate(_to_ef(motion.rotations[0][ix["Hips"]]), FORWARD)
        if math.hypot(f[0], f[1]) > 1e-6:
            a = math.atan2(f[0], f[1])
            yaw = Q.from_axis_angle((0.0, 0.0, 1.0), a)
            report.turned_degrees = math.degrees(a)

    soma_pos = []
    for fr in range(n):
        p = motion.positions(fr, SOMA_HIPS + SOMA_FEET)
        pelvis = Q.scale3(Q.add3(p["LeftLeg"], p["RightLeg"]), 0.5)
        soma_pos.append((pelvis, [p[k][1] for k in SOMA_FEET]))

    stature_ref = _stature(motion)
    leg_ref = _leg_length(motion)
    size = 1.0
    if opts.subject_height and opts.subject_height > 0.0:
        size = opts.subject_height / stature_ref
    elif motion.kind != "bvh":
        est = _estimate_size(motion, [(pel[1], feet) for pel, feet in soma_pos])
        if est is None:
            report.notes.append("could not estimate the person's height (the clip is not "
                                "grounded); assumed %.2f m - set it if strides look off"
                                % stature_ref)
        else:
            size = est
            report.height_estimated = True
    report.subject_height = size * stature_ref
    ef_leg = LENGTHS["Thigh_R"] + LENGTHS["Leg_R"]
    scale = ef_leg / (size * leg_ref)
    report.leg_scale = scale
    stride = opts.stride if opts.stride and opts.stride > 0.0 else scale

    rest = solver.rest
    root = armature.root
    root_head = rest.head[root]
    first = Q.rotate(yaw, Q.rotate(SOMA_TO_EF, soma_pos[0][0]))
    worlds, roots = [], []
    for fr in range(n):
        world = solver.pose(fr, yaw)
        pel = Q.rotate(yaw, Q.rotate(SOMA_TO_EF, soma_pos[fr][0]))
        x, y = pel[0], pel[1]
        if opts.start_at_origin:
            x, y = x - first[0], y - first[1]
        if opts.root_motion == ROOT_IN_PLACE:
            x = y = 0.0
        z = root_head[2] + scale * (pel[2] - size * leg_ref)
        worlds.append(world)
        roots.append((root_head[0] + stride * x, root_head[1] + stride * y, z))

    if opts.ground:
        lows = [min(solver.foot_heights(w, r)) for w, r in zip(worlds, roots)]
        floor = _quantile(lows, 0.02)
        report.floor_offset = -floor
        roots = [(r[0], r[1], r[2] - floor) for r in roots]
    report.hinge_error_degrees = solver.hinge_error

    doc = _document(armature, rest, worlds, roots, motion.fps, opts)
    return doc, report


def _document(armature, rest, worlds, roots, fps, opts):
    times = [time_from_frame(f, fps, EF_TIME_DECIMALS) for f in range(len(worlds))]
    keep = [i for i in range(len(times)) if i == 0 or times[i] > times[i - 1]]
    seams = {s.marker: s for s in SEAMS if s.marker in armature and s.lower in armature}
    tracks = []
    for name in rest.order:
        parent = rest.parent[name]
        driven = name in worlds[0]
        if not driven and not (opts.markers and name in seams) and not opts.static_tracks:
            continue
        keys, prev = [], None
        for i in keep:
            world = worlds[i]
            if name in seams and opts.markers:
                fold = _x_angle(_local(rest, world, seams[name].lower))
                trs = TRS((0.0, 0.0, seam_slide(seams[name], fold)), Q.IDENTITY,
                          (1.0, 1.0, 1.0))
            elif not driven:
                trs = TRS((0.0, 0.0, 0.0), Q.IDENTITY, (1.0, 1.0, 1.0))
            elif parent is None:
                inv = Q.conj(rest.rot[name])
                loc = Q.rotate(inv, Q.sub3(roots[i], rest.head[name]))
                trs = TRS(loc, Q.normalize(Q.mul(inv, world[name])), (1.0, 1.0, 1.0))
            else:
                trs = TRS((0.0, 0.0, 0.0), _local(rest, world, name), (1.0, 1.0, 1.0))
            q = trs.rot if prev is None else Q.align(trs.rot, prev)
            prev = q
            keys.append(quantize_trs(TRS(trs.loc, q, trs.sca), EF_VALUE_DECIMALS))
        tracks.append(Track(name, [times[i] for i in keep], keys, ATTRIBUTES))
    return AnimationDocument(tracks, None, ATTRIBUTES, {}, ("format", "animation"))


def _local(rest, world, name):
    """Delta of `name` against its rest, in its parent's frame: what a pose bone keys."""
    parent = rest.parent[name]
    pw = world.get(parent) if parent is not None else Q.IDENTITY
    if pw is None:
        pw = Q.IDENTITY
    return Q.normalize(Q.mul(Q.conj(rest.local_rot[name]), Q.mul(Q.conj(pw), world[name])))


def convert_file(path, armature: Armature, options=None, fps=None, which="global"):
    """Read any supported file and retarget it. (doc, report)"""
    motion = load_motion(path, fps=fps, which=which)
    return retarget(motion, armature, options)
