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

Reaching arms are straightened. A single camera reads depth worst of anything, and a
punch shows it: the arm is straight in the video while the estimate tips the forearm
towards or away from the lens, folding its length into the elbow. With GEM-X's camera-space
copy of the body at hand, a raised arm's elbow opens to the bend its forearm would have at
the depth that bends it least, on-screen direction kept (see _Solver._depth); then any bend
left under EXTEND_BELOW degrees is opened further (see _extend). Both only open the hinge:
the plane it folds in, and so the arm's twist, stays the one the estimate gave.

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
from .keyfit import fit_indices
from .rigdef import LENGTHS
from .rigedit import SEAMS, seam_slide
from .soma import SomaError, SomaMotion, SomaSkeleton, load_motion

__all__ = ["RetargetOptions", "RetargetReport", "retarget", "convert_file",
           "SOMA_TO_EF", "BODY_SOURCES", "LIMBS", "ROOT_FULL", "ROOT_IN_PLACE",
           "HINGE_FADE", "required_joints", "check_armature", "SomaError",
           "KEYS_ALL", "KEYS_SHARED", "KEYS_PER_BONE", "reduce_tracks",
           "CLAVICLE_MATCH", "CameraTrack"]

SOMA_TO_EF = Q.from_rows(((-1.0, 0.0, 0.0),
                          (0.0, 0.0, 1.0),
                          (0.0, 1.0, 0.0)))

ROOT_FULL = "FULL"
ROOT_IN_PLACE = "IN_PLACE"

KEYS_ALL = "ALL"
KEYS_SHARED = "SHARED"
KEYS_PER_BONE = "PER_BONE"

LOC_PER_DEGREE = 0.005

CLAVICLE_MATCH = 0.4

CAMERA_FLIP = (0.0, 1.0, 0.0, 0.0)

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
    smoothing       gaussian sigma in source frames, 0 off (the default). GEM-X regresses
                    a whole clip at once and is already steady; a filter shaves the peaks
                    off fast moves - a 24 fps punch lasts two or three frames
    root_motion     FULL keeps the path, IN_PLACE keeps only the height
    face_forward    turn the clip so its first frame faces the rig's front
    start_at_origin slide it so its first frame stands on the rig's origin
    ground          drop it so the biped's feet touch z = 0
    subject_height  the filmed person's height in metres, 0 estimates it
    stride          horizontal scale, 0 is the leg-length ratio
    hinge           solve elbows and knees as hinges (see the module docstring)
    tools           give the wrist's orientation to Tool_R / Tool_L
    clavicle        how much of the collarbones' turn the shoulders take, 0..1. The
                    biped's Shoulder bone runs 0.394 m from mid-chest to the arm root, a
                    collarbone 0.155 m, so a full share swings the arm root 2.5 times as
                    far as the person's moved; CLAVICLE_MATCH moves it as far
    hands           0 copies the arms' angles; 1 draws a hand the person holds in front
                    of their body in towards the biped's midline by the shoulder width
                    the person does not have (see _BodyMap) and solves the arm to reach
                    it. The biped's arm roots are 2.3 times as far apart as a person's,
                    so copied angles turn hands held at the chest into elbows jabbed out
                    at the sides
    extend          0..1, how far to straighten a reaching arm - one raised REACH_RAISE
                    degrees or more from the torso, as in a punch; a hanging or swinging
                    arm is left alone. With the camera-space body (incam) the forearm's
                    depth is re-chosen as the straightest the video allows (_depth), and
                    an elbow still bent under EXTEND_BELOW degrees is opened (_extend)
    markers         write Elbow_* / Knee_* seam tracks - for a json going straight into
                    the game; the Blender rig slides them itself
    static_tracks   write rest tracks for joints nothing drives, as shipped clips do
    keys            ALL keeps a key on every frame; PER_BONE lets each track keep only
                    the keys a straight line between its neighbours cannot recreate, which
                    is what the game replays exactly; SHARED keeps the same frames on every
                    track. The Blender import keys every frame and fits smooth curves itself
                    (efbpy.keyposes); this is for a json going straight to the game
    key_tolerance   how far, in degrees, the reduced clip may stray from the capture
                    between keys (the Root and the markers get LOC_PER_DEGREE metres per
                    degree). A bound, not an average: every dropped frame is inside it
    """
    fps: float = 0.0
    start: int = 0
    end: int = -1
    smoothing: float = 0.0
    root_motion: str = ROOT_FULL
    face_forward: bool = True
    start_at_origin: bool = True
    ground: bool = True
    subject_height: float = 0.0
    stride: float = 0.0
    hinge: bool = True
    tools: bool = True
    clavicle: float = 0.4
    hands: float = 1.0
    extend: float = 1.0
    markers: bool = False
    static_tracks: bool = False
    keys: str = KEYS_ALL
    key_tolerance: float = 0.5


@dataclass
class CameraTrack:
    """The video camera, per output frame, in the rig's armature space and at the rig's
    scale: stand a Blender camera here with this lens and the biped covers the person the
    way GEM-X's mesh covers them in its incam video. `rotation` is a Blender camera's
    (looking down -Z, Y up), (w, x, y, z)."""
    location: list
    rotation: list
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def width(self) -> int:
        return int(round(2.0 * self.cx))

    @property
    def height(self) -> int:
        return int(round(2.0 * self.cy))


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
    arms_straightened: int = 0
    keys_dense: int = 0
    keys_kept: int = 0
    key_frames: int = 0
    camera: "CameraTrack | None" = None
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (self.frames_out - 1) / self.fps_out if self.fps_out and self.frames_out else 0.0

    def summary(self) -> str:
        out = ("%d frames at %g fps (%.2f s), subject %.2f m%s, legs x%.3f"
               % (self.frames_out, self.fps_out, self.duration, self.subject_height,
                  " (estimated)" if self.height_estimated else "", self.leg_scale))
        if self.keys_kept and self.keys_kept < self.keys_dense:
            out += ", %d of %d keys kept" % (self.keys_kept, self.keys_dense)
            if self.key_frames:
                out += " on %d key poses" % self.key_frames
        if self.arms_straightened:
            out += ", %d reaching arm frames straightened" % self.arms_straightened
        return out


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


class _BodyMap:
    """Where the biped's hands go when the person's are held in towards their own middle.

    The two bodies disagree about one thing that angles cannot carry: width. A person's
    arm roots sit 0.16 m either side of the spine, the biped's 0.375 m - 2.3 times as far
    apart, with arms no longer - so copied angles leave every hand the person brings in
    front of their chest a hand's width out at the biped's side, which reads as an elbow
    jabbed out. Height and depth agree well enough with the angles and are left to them;
    across the body, a hand inside the person's shoulder line is drawn in by the part of
    the biped's shoulder width the person does not have - all of it at the midline, none
    at the shoulder, so the map is continuous and leaves an arm out at the side alone.
    """

    def __init__(self, motion: SomaMotion, rest: "_Rest"):
        pos = _tpose_positions(motion)
        ix = motion.skeleton.index

        def ef(name):
            return Q.rotate(SOMA_TO_EF, pos[ix[name]])

        palm = Q.scale3(Q.add3(ef("RightHand"), ef("RightHandMiddle1")), 0.5)
        arm_h = (Q.norm3(Q.sub3(ef("RightForeArm"), ef("RightArm")))
                 + Q.norm3(Q.sub3(palm, ef("RightForeArm"))))
        self.anchor = Q.sub3(Q.scale3(Q.add3(pos[ix["RightArm"]], pos[ix["LeftArm"]]), 0.5),
                             pos[ix["Chest"]])
        self.human = abs(ef("RightArm")[0] - ef("LeftArm")[0]) * 0.5
        head = rest.head
        self.biped = abs(head["Arm_R"][0] - head["Arm_L"][0]) * 0.5
        self.upper = Q.norm3(rest.local_t["Hand_R"])
        self.lower = Q.norm3(rest.local_t["Tool_R"])
        self.reach = (self.upper + self.lower) / arm_h
        self.excess = max(0.0, self.biped - self.human * self.reach)

    def pull(self, across):
        """How far in to draw a hand `across` metres out from the person's midline on its
        own side (negative once it has crossed)."""
        share = (self.human - across) / self.human
        return self.excess * min(1.0, max(0.0, share))


SOFT_REACH = 0.08

EXTEND_BELOW = 90.0

REACH_RAISE = (45.0, 70.0)

DEPTH_BEND = (60.0, 100.0)

DEPTH_REPORT = 5.0


def _ramp(value, lo, hi):
    t = min(1.0, max(0.0, (value - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def _degrees_between(a, b):
    return math.degrees(math.atan2(Q.norm3(Q.cross3(a, b)), Q.dot3(a, b)))


def _extend(phi, amount):
    """Straighten a reaching arm: an elbow bent less than EXTEND_BELOW degrees is opened
    along T * x^2 (2 - x), which meets the identity at T with the same slope, so nothing at
    or past a right angle moves and nothing kinks where the curve hands over.

    The second of two steps. _Solver._depth first takes out the bend the estimate put in
    depth, where the camera-space body is there to tell depth from the screen; this opens
    what is left, and is the only step for a file without it (BVH). At full strength 27
    opens to 14, 45 to 34, 60 to 53; a guard at 90 and past is untouched.
    """
    bend = abs(phi)
    t = math.radians(EXTEND_BELOW)
    if bend >= t or amount <= 0.0:
        return phi
    x = bend / t
    opened = t * x * x * (2.0 - x)
    return math.copysign(bend + (opened - bend) * min(1.0, amount), phi)


def _soft(dist, reach, had):
    """Soft IK: stretch past what the arm already had (`had`) and past the last
    SOFT_REACH of its reach is approached asymptotically, so a target hovering at full
    stretch cannot flick the elbow between bent and locked - and a target the copied angles
    already reach is reached exactly, which keeps the pull continuous from zero."""
    knee = max(reach * (1.0 - SOFT_REACH), min(had, reach - 1e-6))
    if dist <= knee:
        return dist
    room = reach - knee
    return reach - room * math.exp(-(dist - knee) / room)


class _Solver:
    def __init__(self, motion: SomaMotion, armature: Armature, opts: RetargetOptions,
                 incam: SomaMotion | None = None):
        self.m = motion
        self.cam = incam
        self.sk = motion.skeleton
        self.rest = _rest(armature)
        self.opts = opts
        ix = self.sk.index
        self.ix = ix
        self.body = _BodyMap(motion, self.rest) if opts.hands > 0.0 else None
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
        self.straightened = 0

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
            keep, open_by = 1.0, 0.0
            if limb.tool and opts.extend > 0.0:
                open_by = min(1.0, opts.extend) * self._reaching(frame, yaw, g_up, limb)
                if open_by > 0.0 and self.cam is not None:
                    keep = self._depth(frame, yaw, limb, g_up, g_low, open_by)
            up = Q.normalize(Q.mul(g_up, Q.mul(self.align[limb.upper], rest.rot[limb.upper])))
            low = Q.normalize(Q.mul(g_low, Q.mul(self.align[limb.lower],
                                                 rest.rot[limb.lower])))
            if self.body is not None and limb.tool:
                up, low = self._reach(frame, yaw, limb, out, up, low, g_low, keep, open_by)
            elif opts.hinge:
                up, low = self._hinge(limb, up, Q.rotate(g_low, self.src_dir[limb.lower]),
                                      keep, open_by)
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

    def _reaching(self, frame, yaw, g_up, limb):
        """How much an arm is reaching, 0..1: its upper arm's angle from the torso's down
        axis, ramped over REACH_RAISE. Punches measure 70-105 degrees, a walk's swing
        10-50, so straightening reaches cannot straighten a walk."""
        u = Q.rotate(g_up, self.src_dir[limb.upper])
        down = Q.rotate(self.g(frame, "Chest", yaw), (0.0, 0.0, -1.0))
        return _ramp(_degrees_between(u, down), *REACH_RAISE)

    def _ray(self, frame, yaw, joint):
        """The line of sight to `joint`, armature space: the camera-space body puts the
        lens at the origin, and the Hips' two rotations turn it into the world's frame."""
        hips = self.ix["Hips"]
        to_cam = Q.mul(self.cam.rotations[frame][hips], Q.conj(self.m.rotations[frame][hips]))
        seen = self.cam.positions(frame, (joint,))[joint]
        return Q.rotate(Q.mul(yaw, SOMA_TO_EF), Q.rotate(Q.conj(to_cam), Q.unit3(seen)))

    def _depth(self, frame, yaw, limb, g_up, g_low, amount):
        """How much of a reaching elbow's bend the video bears out, 0..1.

        What a video shows of a forearm is its direction across the frame; how far it tips
        towards or away from the lens is the estimate's guess, and at a punch's peak the
        guess is where the bend comes from - the forearm tipped 40-70% of its length
        towards the camera, the upper arm flat. So the forearm is kept in the plane of its
        on-screen direction and the line of sight, and turned within it towards the
        direction there that is closest to the upper arm: the straightest arm the video
        allows. Only up to DEPTH_BEND degrees of bend - a guard is folded whatever the
        depth - and only as far as the upper arm agrees on screen with the forearm.

        What comes back is the ratio of that bend to the estimate's, for the hinge to fold
        by; the forearm is not turned onto the new direction. As the arm straightens, the
        plane through the upper arm and the new direction swings round - near straight it
        is any plane at all - and the hinge, which turns the upper arm to fold in the
        limb's plane, would spin the arm about itself with it: 55-77 degrees in a frame at
        a punch. The estimate's own plane is steady, so the arm keeps its twist."""
        u = Q.rotate(g_up, self.src_dir[limb.upper])
        v = Q.unit3(Q.rotate(g_low, self.src_dir[limb.lower]))
        bend = _degrees_between(u, v)
        w = amount * (1.0 - _ramp(bend, *DEPTH_BEND))
        if w <= 0.0 or bend < 1e-3:
            return 1.0
        r = self._ray(frame, yaw, limb.lower_src[0])
        flat = Q.sub3(v, Q.scale3(r, Q.dot3(v, r)))
        if Q.norm3(flat) < 1e-6:
            return 1.0
        p = Q.unit3(flat)
        along = Q.dot3(u, p) / max(1e-9, Q.norm3(u))
        w *= _ramp(along, 0.0, 0.5)
        if w <= 0.0:
            return 1.0
        best = Q.unit3(Q.add3(Q.scale3(p, Q.dot3(u, p)), Q.scale3(r, Q.dot3(u, r))))
        want = Q.unit3(Q.add3(Q.scale3(v, 1.0 - w), Q.scale3(best, w)), v)
        opened = min(bend, _degrees_between(u, want))
        if bend - opened > DEPTH_REPORT:
            self.straightened += 1
        return opened / bend

    def _reach(self, frame, yaw, limb, out, up, low, g_low, keep=1.0, open_by=0.0):
        """Draw the hand in towards the midline by the body map and re-solve the arm onto
        it, keeping the person's swivel. Returns (upper, lower) world rotations.

        The hand the copied angles give is the starting point, so height, depth and every
        pose with the hand out at the side come through untouched; only the across-the-body
        position moves, and the arm is re-solved - two bones, the elbow turned the way the
        person's points - to reach it.
        """
        rest, body, opts = self.rest, self.body, self.opts
        side = "Right" if limb.upper.endswith("_R") else "Left"
        outward = 1.0 if side == "Right" else -1.0
        carry = Q.mul(yaw, SOMA_TO_EF)
        p = self.m.positions(frame, ("Chest", side + "Arm", side + "ForeArm", side + "Hand",
                                     side + "HandMiddle1"))
        palm = Q.scale3(Q.add3(p[side + "Hand"], p[side + "HandMiddle1"]), 0.5)
        anchor = Q.add3(p["Chest"], Q.rotate(self.m.rotations[frame][self.ix["Chest"]],
                                             body.anchor))
        g_chest = self.g(frame, "Chest", yaw)
        across = outward * Q.rotate(Q.conj(g_chest), Q.rotate(carry, Q.sub3(palm, anchor)))[0]
        pull = body.pull(across) * min(1.0, opts.hands)
        if pull <= 1e-6:
            if opts.hinge:
                return self._hinge(limb, up, Q.rotate(g_low, self.src_dir[limb.lower]),
                                   keep, open_by)
            return up, low

        shoulder = limb.upper.replace("Arm", "Shoulder")
        root = Q.add3(Q.rotate(out["Chest"], rest.local_t[shoulder]),
                      Q.rotate(out[shoulder], rest.local_t[limb.upper]))
        elbow_fk = Q.add3(root, Q.rotate(up, rest.local_t[limb.lower]))
        grip_fk = Q.add3(elbow_fk, Q.rotate(low, rest.local_t[limb.tool]))
        target = Q.sub3(grip_fk, Q.rotate(g_chest, (outward * pull, 0.0, 0.0)))

        a, b = body.upper, body.lower
        d = Q.sub3(target, root)
        dist = max(abs(a - b) + 1e-3,
                   _soft(Q.norm3(d), a + b, Q.norm3(Q.sub3(grip_fk, root))))
        e1 = Q.unit3(d)
        bend = Q.sub3(elbow_fk, root)
        swivel = Q.rotate(carry, Q.sub3(p[side + "ForeArm"], p[side + "Arm"]))
        pole = Q.sub3(swivel, Q.scale3(e1, Q.dot3(swivel, e1)))
        if Q.norm3(pole) < 0.02:
            pole = Q.sub3(bend, Q.scale3(e1, Q.dot3(bend, e1)))
        e2 = Q.unit3(pole, (0.0, -1.0, 0.0))
        cos_a = max(-1.0, min(1.0, (a * a + dist * dist - b * b) / (2.0 * a * dist)))
        sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
        elbow = Q.add3(root, Q.scale3(Q.add3(Q.scale3(e1, cos_a), Q.scale3(e2, sin_a)), a))
        grip = Q.add3(root, Q.scale3(e1, dist))

        up = Q.normalize(Q.mul(Q.between(Q.rotate(up, (0.0, 1.0, 0.0)),
                                         Q.sub3(elbow, root)), up))
        want = Q.sub3(grip, elbow)
        if opts.hinge:
            return self._hinge(limb, up, want, keep, open_by)
        low = Q.normalize(Q.mul(Q.between(Q.rotate(low, (0.0, 1.0, 0.0)), want), low))
        return up, low

    def _hinge(self, limb, up, want_low_dir, keep=1.0, open_by=0.0):
        """Turn the upper bone about itself until the limb's bend lies in its hinge plane,
        then fold the lower bone about the hinge alone - by `keep` of the bend, opened
        further by _extend(open_by)."""
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
        if open_by > 0.0:
            reach = _extend(phi * keep, open_by)
            if reach != phi:
                fold = reach - self.rest_fold[limb.lower]
                local = Q.mul(rest.local_rot[limb.lower],
                              Q.from_axis_angle((1.0, 0.0, 0.0), fold))
                low = Q.normalize(Q.mul(up, local))
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


def retarget(motion: SomaMotion, armature: Armature, options=None, incam=None,
             intrinsics=None):
    """(AnimationDocument, RetargetReport) for `motion` on `armature`.

    `incam` is the same clip in camera space, GEM-X's body_params_incam: it tells a
    reaching arm's depth apart from its direction on screen (_Solver._depth), and with
    the camera `intrinsics` the report also carries the video camera as a CameraTrack.
    """
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
    if incam is not None and not flipped:
        incam = _window(incam, opts.start, opts.end)
        incam = _smooth(incam, float(opts.smoothing or 0.0), joints)
        incam = _resample(incam, motion.fps, joints)
    else:
        incam = None
    n = motion.frame_count
    if n == 0:
        raise SomaError("no frames left to convert")
    report.frames_out, report.fps_out = n, motion.fps

    solver = _Solver(motion, armature, opts,
                     incam if incam is not None and incam.frame_count == n else None)
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
    report.arms_straightened = solver.straightened
    if incam is not None and incam.frame_count == n and intrinsics:
        report.camera = _camera(motion, incam, intrinsics, yaw, scale,
                                [p for p, _feet in soma_pos], roots)

    doc = _document(armature, rest, worlds, roots, motion.fps, opts)
    report.keys_dense = sum(len(t) for t in doc.tracks)
    doc.tracks, report.key_frames = reduce_tracks(doc.tracks, opts.keys, opts.key_tolerance)
    report.keys_kept = sum(len(t) for t in doc.tracks)
    return doc, report


def _camera(world, incam, intrinsics, yaw, scale, pelvis, roots) -> CameraTrack:
    """Where the phone was, frame by frame, carried into the rig's space.

    The two parameter sets are one body seen from two frames, so the Hips' rotation and
    position in each give the world-to-camera transform outright:
        R = G_incam[Hips] @ G_world[Hips]^-1      camera = hips_world - R^-1 hips_incam
    The camera is then placed against the pelvis rather than the world, and at the rig's
    scale, so it frames the biped exactly as the clip's own root placement - in place,
    turned to face forward, dropped onto the floor - leaves it.
    """
    hips = world.skeleton.index["Hips"]
    carry = Q.mul(yaw, SOMA_TO_EF)
    loc, rot = [], []
    for f in range(world.frame_count):
        r = Q.normalize(Q.mul(incam.rotations[f][hips], Q.conj(world.rotations[f][hips])))
        cam = Q.sub3(world.hips[f], Q.rotate(Q.conj(r), incam.hips[f]))
        offset = Q.rotate(carry, Q.scale3(Q.sub3(cam, pelvis[f]), scale))
        loc.append(Q.add3(roots[f], offset))
        rot.append(Q.normalize(Q.mul(Q.mul(carry, Q.conj(r)), CAMERA_FLIP)))
    for i in range(1, len(rot)):
        rot[i] = Q.align(rot[i], rot[i - 1])
    fx, fy, cx, cy = intrinsics
    return CameraTrack(loc, rot, fx, fy, cx, cy)


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


def _key_rows(track, tol_deg):
    """One row per key, each channel divided by its own tolerance so that 1.0 is the bound
    everywhere. A quaternion component within theta/4 of the line keeps the whole rotation
    within theta; a location within its bound over each axis stays within it."""
    rot = math.radians(tol_deg) / 4.0
    loc = tol_deg * LOC_PER_DEGREE / 2.0
    moves = any(any(abs(v) > 1e-9 for v in k.loc) for k in track.keys)
    rows = []
    for k in track.keys:
        row = [v / rot for v in k.rot]
        if moves:
            row += [v / loc for v in k.loc]
        rows.append(row)
    return rows


def reduce_tracks(tracks, mode=KEYS_SHARED, tol_deg=1.0):
    """(tracks, shared key count) with only the keys linear interpolation cannot recreate.

    Both Blender's LINEAR curves and Epic Fight's lerp / nlerp draw straight lines between
    keys, so a dropped key costs exactly its distance from the line its neighbours draw -
    which is what efb.keyfit bounds. SHARED fits the whole body at once and keeps one set of
    frames for every track; PER_BONE fits each track alone. The ends always stay, so every
    track keeps at least two keys, as every shipped one has.
    """
    if mode == KEYS_ALL or tol_deg <= 0.0 or not tracks:
        return tracks, 0
    if mode == KEYS_SHARED:
        times = tracks[0].times
        if any(t.times != times for t in tracks):
            mode = KEYS_PER_BONE
        else:
            per = [_key_rows(t, tol_deg) for t in tracks]
            rows = [[v for p in per for v in p[i]] for i in range(len(times))]
            keep = fit_indices(times, rows, 1.0)
            return [Track(t.name, [t.times[i] for i in keep], [t.keys[i] for i in keep],
                          t.format) for t in tracks], len(keep)
    out = []
    for t in tracks:
        keep = fit_indices(t.times, _key_rows(t, tol_deg), 1.0)
        out.append(Track(t.name, [t.times[i] for i in keep], [t.keys[i] for i in keep],
                         t.format))
    return out, 0


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
