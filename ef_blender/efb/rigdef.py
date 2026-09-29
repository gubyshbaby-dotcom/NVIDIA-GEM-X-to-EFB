"""The Epic Fight biped rig, emitted as data. No bpy.

Everything the bpy layer needs is a table: bones with armature-space rest matrices,
constraints, drivers, custom properties, bone collections, widget assignments and the
IK/FK snap wiring. `validate` is the headless gate over all of it.

The 20 Epic Fight joints keep their exact names and rest matrices and are the only
deforming bones. Controls drive them through constraints; nothing a control does can
move a deform bone's rest pose, so the exporter still reads a faithful EF skeleton.

One consequence for the exporter: deform bones are constrained, so pose_bone.matrix_basis
is NOT the Epic Fight delta any more. Use the constraint-aware form,
    delta = inv(rest_local) @ (parent_pose.matrix.inverted() @ pose_bone.matrix)
which is the identity blender-ground section 3 lists second.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace

from .armature import Armature
from .matrix import Mat4
from .rigedit import HINGE, TOOL_WIDGET_TURN
from .rigmath import add, cross, dot, length, normalize, pole_angle, scale, sub
from .widgets import WIDGETS

__all__ = ["RigBone", "RigConstraint", "RigDriver", "RigCollectionDriver", "DriverVar",
           "RigFlagDriver", "DistanceVar",
           "RigProperty", "RigCollection", "IkLimb", "RigDefinition", "build_biped_rig",
           "validate", "limbs_to_json", "limbs_from_json", "RIG_ID", "RIG_VERSION",
           "ADDON_VERSION",
           "DEFAULT_IK", "SHOW_INACTIVE", "ROTATOR", "HANDLE", "PIVOT_EXCEPTIONS",
           "COLOURS", "UNCOLOURED", "FOLD_SIGN", "IK_SEED", "HINGE_LOCK", "ik_band",
           "seed_chain",
           "LOCK_LIMB_ROTATION", "GLOBAL_HEAD_ROTATION",
           "end_lock_prop", "mirror_across",
           "SMOOTH_DEFORM", "SUBDIVISION",
           "POLE_REFERENCE", "POLE_FOLLOW", "pole_follow_prop", "POLE_FOLLOW_SPACE",
           "POLE_ROOT_SPACE", "POLE_FOLLOW_MEASURED", "DEFAULT_POLE_FOLLOW",
           "TOOL_BIND", "tool_bind_prop", "TOOL_JOINTS", "TOOL_BIND_CONSTRAINT",
           "tool_holder", "TOOL_KEYED", "tool_keyed_prop",
           "DETACH_COPY", "detach_ctrl", "detach_twin"]

RIG_ID = "efb_biped"
RIG_VERSION = 12

ADDON_VERSION = "0.25.0"

CTRL = "CTRL-"
FK = "FK-"
IK = "IK-"
POLE = "POLE-"
MCH = "MCH-"

DETACH_COPY = "Detach"


def detach_ctrl(upper: str) -> str:
    """The control that offsets a whole limb off the torso, at its root joint."""
    return CTRL + "Detach-" + upper


def detach_twin(upper: str) -> str:
    """The hidden twin under the master that replays the detach control's LOCAL pose.

    The handle has to ride the detach without riding the spine: the control hangs off the
    limb's socket joint so the FK and solver chains under it follow the body as they always
    did, but a Child Of aimed at it would drag the spine's motion onto the handle too. The
    twin rests exactly on the control, hangs off the master, and copies the control's
    matrix_basis LOCAL to LOCAL - so it moves by the detach offset conjugated about the
    limb root, and by nothing else. With the detach at rest it IS the master's carry."""
    return MCH + "Detach-" + upper

LENGTHS = {
    "Root": 0.05, "Torso": 0.3, "Chest": 0.4, "Head": 0.5,
    "Shoulder_R": 0.3939, "Arm_R": 0.3, "Hand_R": 0.2729, "Tool_R": 0.15,
    "Shoulder_L": 0.3939, "Arm_L": 0.3, "Hand_L": 0.2729, "Tool_L": 0.15,
    "Thigh_R": 0.3747, "Leg_R": 0.3864, "Thigh_L": 0.3747, "Leg_L": 0.3864,
    "Knee_R": 0.15, "Knee_L": 0.15, "Elbow_R": 0.15, "Elbow_L": 0.15,
}

POLE_DISTANCE = {"Elbow_R": 0.38, "Elbow_L": 0.38, "Knee_R": 0.34, "Knee_L": 0.34}

ROTATOR = "rotator"
HANDLE = "handle"

SIZES = {
    "master": 1.50, "cog": 0.18, "root": 0.55, "torso": 0.46, "chest": 0.40,
    "head": 0.62, "shoulder": 0.35, "tool": 0.38, "pole": 0.17,
    "fk_arm": 0.34, "fk_hand": 0.32, "fk_thigh": 0.30, "fk_leg": 0.30,
    "ik_hand": 0.22, "ik_leg": 0.29,
    "detach": 0.16,
}

_ZERO = (0.0, 0.0, 0.0)

PIVOT_EXCEPTIONS = {
    CTRL + "Head": ("the pivot is the skull base, but a ring at the neck cannot say "
                    "'head'. The cage is drawn around the head cube and still contains "
                    "the pivot, which is the reference rig's own treatment."),
}

WIDGET_OFFSETS = {
    CTRL + "Head": ((0.0, LENGTHS["Head"] * 0.572, 0.0), _ZERO),
}
for _joint in ("Tool_R", "Tool_L"):
    WIDGET_OFFSETS[CTRL + _joint] = (_ZERO, TOOL_WIDGET_TURN)
for _side, _out in (("_R", 1.0), ("_L", -1.0)):
    WIDGET_OFFSETS[IK + "Leg" + _side] = ((0.04 * _out, 0.0, 0.0), _ZERO)
    WIDGET_OFFSETS[IK + "Hand" + _side] = ((0.075 * _out, 0.0, 0.0), _ZERO)
    WIDGET_OFFSETS[detach_ctrl("Arm" + _side)] = (_ZERO, (0.0, 0.0, -20.0 * _out))
    WIDGET_OFFSETS[detach_ctrl("Thigh" + _side)] = (_ZERO, (0.0, 0.0, -15.0 * _out))

DEFAULT_IK = {"arm_r": 0.0, "arm_l": 0.0, "leg_r": 1.0, "leg_l": 1.0}

SHOW_INACTIVE = "show_inactive"

LOCK_LIMB_ROTATION = "lock_limb_rotation"

GLOBAL_HEAD_ROTATION = "global_head_rotation"

def end_lock_prop(key: str) -> str:
    """How much the hand or foot takes its roll from the IK control instead of the solve."""
    return ("wrist_lock_" if key.startswith("arm") else "ankle_lock_") + key


TOOL_JOINTS = ("Tool_R", "Tool_L")

TOOL_BIND = "tool_bind"

TOOL_BIND_CONSTRAINT = "Bind"


def tool_holder(joint: str) -> str:
    return MCH + CTRL + joint + ".space"


def tool_bind_prop(joint: str) -> str:
    return TOOL_BIND + "_" + joint[-1].lower()


TOOL_KEYED = "tool_keyed"


def tool_keyed_prop(joint: str) -> str:
    return TOOL_KEYED + "_" + joint[-1].lower()


POLE_FOLLOW = "pole_follow"

DEFAULT_POLE_FOLLOW = 1.0

POLE_FOLLOW_SPACE = "Chord"
POLE_ROOT_SPACE = "Root"

POLE_FOLLOW_MEASURED = {"chord": 0.82, "world": 29.37, "poses": 4518}


def pole_follow_prop(key: str) -> str:
    return POLE_FOLLOW + "_" + key


POLE_REFERENCE = {
    "arm_r": (0.420, 0.719, 0.553),
    "arm_l": (-0.420, 0.719, 0.553),
    "leg_r": (0.821, 0.472, -0.321),
    "leg_l": (-0.821, 0.472, -0.321),
}

POLE_REF_ARM = 0.3

SMOOTH_DEFORM = "smooth_deform"

SUBDIVISION = 14

ROLL_ARM = 0.3

FOLD_SIGN = {"arm_r": 1, "arm_l": 1, "leg_r": -1, "leg_l": -1}

IK_SEED = {"arm_r": 8.0, "arm_l": 8.0, "leg_r": 0.5, "leg_l": 0.5}

HINGE_LOCK = (False, True, True)


def ik_band(key: str) -> tuple:
    """Degrees of local X the solver's hinge may travel, relative to the DEFORM rest.

    The limb's whole fold band, the same one the panel prints and the FK control and the
    export stop carry. A rig generated at this band and a panel never touched agree from
    the first frame.

    This used to drop the hyperextension half - (0, hi) on an arm - on the grounds that a
    negative fold under IK puts the elbow on the wrong side of the shoulder-to-wrist line.
    Swept over 451 handle placements per limb, 11 directions from 30 to 130 percent of the
    chain's span: the solver never chose a fold on the wrong side either way, the reachable
    fold bottomed out at +6.51 on the arm and -0.07 on the leg, and widening the band to
    the full panel range moved no solve by more than 0.000001 m. The seed and HINGE_LOCK
    are what hold the side; the floor only made the panel print a number nobody enforced.
    """
    _axis, lo, hi, _on, _slack = HINGE["arm" if key.startswith("arm") else "leg"]
    return (lo, hi)

COLOURS = {
    "Root": "THEME09", "Spine": "THEME03",
    "FK Arm.R": "THEME01", "FK Arm.L": "THEME04",
    "FK Leg.R": "THEME01", "FK Leg.L": "THEME04",
    "IK Arm.R": "THEME06", "IK Arm.L": "THEME05",
    "IK Leg.R": "THEME06", "IK Leg.L": "THEME05",
    "Detach": "THEME07", "Sockets": "THEME02",
    "Deform": "THEME08", "Mechanism": "DEFAULT",
}

UNCOLOURED = "Mechanism"


@dataclass(frozen=True)
class RigBone:
    name: str
    rest: Mat4
    length: float
    parent: str | None = None
    collection: str = "Mechanism"
    use_deform: bool = False
    hide: bool = False
    widget: str = ""
    widget_size: float = 1.0
    widget_translation: tuple = (0.0, 0.0, 0.0)
    widget_rotation: tuple = (0.0, 0.0, 0.0)
    role: str = ""
    ef_joint: str | None = None
    lock_location: bool = False
    lock_select: bool = False
    rotation_mode: str = "QUATERNION"
    ik_lock: tuple = ()
    ik_limit: tuple = ()


@dataclass(frozen=True)
class RigConstraint:
    owner: str
    name: str
    type: str
    subtarget: str = ""
    influence: float = 1.0
    chain_count: int = 0
    pole_subtarget: str = ""
    pole_angle: float = 0.0
    use_rotation: bool = False
    use_stretch: bool = False
    track_axis: str = ""
    lock_axis: str = ""
    distance: float = 0.0
    limit_mode: str = ""
    use_axis: tuple = ()
    target_space: str = ""
    owner_space: str = ""
    seat_inverse: bool = False


@dataclass(frozen=True)
class DriverVar:
    name: str
    prop: str


@dataclass(frozen=True)
class DistanceVar:
    """A driver variable holding the gap between two bone heads, in metres. This is how a
    stretch knows it is out of reach without anything having to measure it in python."""
    name: str
    bone_a: str
    bone_b: str


@dataclass(frozen=True)
class RigDriver:
    bone: str
    constraint: str
    expression: str
    variables: tuple

    @property
    def data_path(self) -> str:
        return 'pose.bones["%s"].constraints["%s"].influence' % (self.bone,
                                                                 self.constraint)


@dataclass(frozen=True)
class RigFlagDriver:
    """A driver on something that is not a constraint influence: a per-bone IK stretch
    weight, an inherit-rotation flag, a transform lock. `on_data` picks the armature data
    over the object, and `index` names one element of an array property."""
    bone: str
    path: str
    expression: str
    variables: tuple
    on_data: bool = False
    index: int = -1

    @property
    def data_path(self) -> str:
        prefix = 'bones["%s"].' if self.on_data else 'pose.bones["%s"].'
        return (prefix % self.bone) + self.path


@dataclass(frozen=True)
class RigCollectionDriver:
    """Drives a bone collection's visibility off the blend, so the drawn controls are
    exactly the live ones and no operator has to remember to swap them."""
    collection: str
    expression: str
    variables: tuple

    @property
    def data_path(self) -> str:
        return 'collections["%s"].is_visible' % self.collection


@dataclass(frozen=True)
class RigProperty:
    name: str
    kind: str
    default: float
    minimum: float = 0.0
    maximum: float = 1.0
    description: str = ""
    items: tuple = ()


@dataclass(frozen=True)
class RigCollection:
    name: str
    visible: bool = True


@dataclass(frozen=True)
class IkLimb:
    key: str
    label: str
    prop: str
    upper: str
    lower: str
    marker: str
    fk_upper: str
    fk_lower: str
    ik_upper: str
    ik_lower: str
    ik_ctrl: str
    pole_ctrl: str
    fk_collection: str
    ik_collection: str
    pole_angle: float
    ik_twist: str = ""
    default_ik: float = 0.0
    end_lock: str = ""
    end_roll: str = ""
    span: float = 0.0
    pole_space: str = ""
    pole_chord: str = ""
    pole_ref: str = ""
    pole_follow: str = ""
    detach: str = ""
    detach_twin: str = ""
    ik_seed: float = 0.0


@dataclass
class RigDefinition:
    name: str
    armature: Armature
    bones: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    drivers: list = field(default_factory=list)
    flag_drivers: list = field(default_factory=list)
    collection_drivers: list = field(default_factory=list)
    properties: list = field(default_factory=list)
    collections: list = field(default_factory=list)
    limbs: list = field(default_factory=list)
    widgets: dict = field(default_factory=dict)
    rig_id: str = RIG_ID
    version: int = RIG_VERSION

    def bone(self, name) -> RigBone:
        return self._index[name]

    @property
    def _index(self):
        return {b.name: b for b in self.bones}

    def deform_bones(self):
        return [b for b in self.bones if b.ef_joint]

    def limb(self, key) -> IkLimb:
        for l in self.limbs:
            if l.key == key:
                return l
        raise KeyError(key)

    def property_names(self):
        return [p.name for p in self.properties]




def _rot_only(m: Mat4) -> Mat4:
    r = m.rows()
    return Mat4.from_rows(((r[0][0], r[0][1], r[0][2], 0.0),
                           (r[1][0], r[1][1], r[1][2], 0.0),
                           (r[2][0], r[2][1], r[2][2], 0.0),
                           (0.0, 0.0, 0.0, 1.0)))


def _along(m: Mat4, distance: float) -> Mat4:
    """Same orientation, head slid `distance` along the bone's own +Y."""
    return m @ Mat4.translation(0.0, distance, 0.0)


def _x_axis(m: Mat4):
    r = m.rows()
    return (r[0][0], r[1][0], r[2][0])


def _head(m: Mat4):
    return m.to_translation()


def _sideways(m: Mat4, distance: float) -> Mat4:
    """Same orientation, head slid `distance` along the bone's own +X."""
    return m @ Mat4.translation(distance, 0.0, 0.0)


def _bone_axis(m: Mat4):
    r = m.rows()
    return (r[0][1], r[1][1], r[2][1])


def _from_frame(m: Mat4, v):
    """`v`, given in the columns of `m`, as an armature-space direction."""
    r = m.rows()
    return tuple(r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] for i in range(3))


def _frame(y_axis, hint) -> Mat4:
    """A rotation whose +Y is `y_axis` and whose +X is `hint` squared off it."""
    y = normalize(y_axis)
    x = sub(hint, scale(y, dot(hint, y)))
    x = normalize(x) if length(x) > 1e-9 else normalize(cross(y, (0.0, 0.0, 1.0)))
    z = cross(x, y)
    return Mat4.from_rows(((x[0], y[0], z[0], 0.0),
                           (x[1], y[1], z[1], 0.0),
                           (x[2], y[2], z[2], 0.0),
                           (0.0, 0.0, 0.0, 1.0)))


def pole_site(root, middle, tip, marker_axis, distance):
    """Where a limb's pole control belongs: abeam the middle joint, along the heading the
    Epic Fight marker joint already has. Elbow_* points backwards, Knee_* forwards, so an
    elbow pole lands behind the figure and a knee pole in front.

    This picks the SWIVEL only. Which way the joint folds is HINGE_LOCK's job, because a
    pole sits at a fixed point in armature space and the fold direction rotates with the
    shoulder - that mismatch is what inverted 22% of shipped poses.
    """
    span = sub(tip, root)
    out = sub(marker_axis, scale(span, dot(marker_axis, span) / dot(span, span)))
    return add(middle, scale(normalize(out), distance))


def _about(pivot, axis, radians) -> Mat4:
    return (Mat4.translation(*pivot) @ Mat4.rotation_axis(radians, axis)
            @ Mat4.translation(*[-v for v in pivot]))


def mirror_across(m: Mat4, root, tip) -> Mat4:
    """`m` turned half a turn about the root-to-tip line. For the pole that is its other
    side exactly - same distance off the limb, same station along it, opposite half-plane,
    so the fold plane turns over by 180 degrees - and it carries the frame with it, so the
    cone still points back at the joint it steers."""
    return _about(root, normalize(sub(tip, root)), math.pi) @ m


def seed_chain(up_rest: Mat4, up_len: float, low_rest: Mat4, degrees: float):
    """Kinks a two bone chain by `degrees` of the lower bone's own local X - the hinge -
    half at each end, keeping both heads joined and both lengths. Returns the new rests.

    Turning about the lower bone's own X and nothing else is what leaves that X the exact
    hinge axis, which is the axis HINGE_LOCK and ik_band are then expressed in. Bending
    toward the pole instead tilts it by however far the pole is off the joint's plane.
    """
    if not degrees:
        return up_rest, low_rest
    half = math.radians(-degrees * 0.5)
    axis = _x_axis(low_rest)
    pre_up = _about(_head(up_rest), axis, half) @ up_rest
    tilted = _about(_head(low_rest), axis, -half) @ low_rest
    return pre_up, Mat4.translation(*sub(_head(_along(pre_up, up_len)),
                                         _head(low_rest))) @ tilted


def _place(bone: RigBone) -> RigBone:
    """Hands a control its entry out of WIDGET_OFFSETS."""
    t, r = WIDGET_OFFSETS.get(bone.name, (_ZERO, _ZERO))
    if t == _ZERO and r == _ZERO:
        return bone
    return replace(bone, widget_translation=t, widget_rotation=r)


def build_biped_rig(arm: Armature, name: str = "EF-biped") -> RigDefinition:
    rig = RigDefinition(name=name, armature=arm)

    def add(bone):
        rig.bones.append(_place(bone))
    con = rig.constraints.append

    rig.collections.extend([RigCollection("Root"), RigCollection("Spine")])
    for spec in _LIMBS:
        fk_coll, ik_coll = _limb_collections(spec)
        ik = DEFAULT_IK[spec.key]
        rig.collections.append(RigCollection(fk_coll, visible=ik < 1.0))
        rig.collections.append(RigCollection(ik_coll, visible=ik > 0.0))
    rig.collections.extend([
        RigCollection("Detach"),
        RigCollection("Sockets"),
        RigCollection("Deform", visible=False),
        RigCollection("Mechanism", visible=False),
    ])

    rest = {j: arm.rest_model(j) for j in arm.names()}

    for jname in arm.depth_first():
        j = arm[jname]
        add(RigBone(jname, rest[jname], LENGTHS[jname], j.parent, "Deform",
                    use_deform=True, ef_joint=jname))

    add(RigBone(CTRL + "Master", _rot_only(rest["Root"]), 0.25, None, "Root",
                widget="ground", widget_size=SIZES["master"], role=ROTATOR))
    add(RigBone(CTRL + "COG", rest["Root"], 0.2, CTRL + "Master", "Root",
                widget="diamond", widget_size=SIZES["cog"], role=ROTATOR))
    add(RigBone(CTRL + "Root", rest["Root"], 0.15, CTRL + "COG", "Root",
                widget="belt", widget_size=SIZES["root"], role=ROTATOR))
    con(RigConstraint("Root", "FK", "COPY_TRANSFORMS", CTRL + "Root"))

    for jname, size in (("Torso", SIZES["torso"]), ("Chest", SIZES["chest"]),
                        ("Head", SIZES["head"])):
        parent = arm[jname].parent
        add(RigBone(CTRL + jname, rest[jname], LENGTHS[jname], parent, "Spine",
                    widget="collar" if jname != "Head" else "cube", widget_size=size,
                    lock_location=True, role=ROTATOR))
        con(RigConstraint(jname, "FK", "COPY_TRANSFORMS", CTRL + jname))

    rig.flag_drivers.append(RigFlagDriver(
        CTRL + "Head", "use_inherit_rotation", "1 - g",
        (DriverVar("g", GLOBAL_HEAD_ROTATION),), on_data=True))

    for jname in ("Shoulder_R", "Shoulder_L"):
        add(RigBone(CTRL + jname, rest[jname], LENGTHS[jname], arm[jname].parent,
                    "Spine", widget="wedge", widget_size=SIZES["shoulder"],
                    lock_location=True, role=ROTATOR))
        con(RigConstraint(jname, "FK", "COPY_TRANSFORMS", CTRL + jname))

    for spec in _LIMBS:
        _build_limb(rig, arm, rest, spec)

    for jname in TOOL_JOINTS:
        holder = tool_holder(jname)
        add(RigBone(holder, rest[arm[jname].parent], 0.08, None, "Mechanism", hide=True))
        con(RigConstraint(holder, TOOL_BIND_CONSTRAINT, "CHILD_OF", arm[jname].parent,
                          influence=1.0, seat_inverse=True))
        rig.drivers.append(RigDriver(holder, TOOL_BIND_CONSTRAINT, "b",
                                     (DriverVar("b", tool_bind_prop(jname)),)))
        add(RigBone(CTRL + jname, rest[jname], LENGTHS[jname], holder,
                    "Sockets", widget="socket", widget_size=SIZES["tool"], role=ROTATOR))
        con(RigConstraint(jname, "FK", "COPY_TRANSFORMS", CTRL + jname))

    rig.properties.extend(
        [RigProperty(l.prop, "float", l.default_ik, 0.0, 1.0,
                     "0 = FK, 1 = IK for %s. Only the live half is drawn" % l.label)
         for l in rig.limbs]
        + [RigProperty(SHOW_INACTIVE, "bool", 0.0, 0.0, 1.0,
                       "Draw the IK and FK controls that are currently inert"),
           RigProperty("snap_keyframe", "bool", 0.0, 0.0, 1.0,
                       "Key the affected bones on both sides of an IK/FK snap"),
           RigProperty(LOCK_LIMB_ROTATION, "bool", 1.0, 0.0, 1.0,
                       "Hold the elbows and knees to their fold axis. Off, they turn "
                       "freely and an FK <- IK snap stops being exact"),
           RigProperty(GLOBAL_HEAD_ROTATION, "bool", 0.0, 0.0, 1.0,
                       "The head keeps its world orientation while the body turns"),
           RigProperty(SMOOTH_DEFORM, "bool", 0.0, 0.0, 1.0,
                       "Draw a stylised body whose limbs bend as one piece. It is NOT "
                       "what the game draws - up to 0.20 m off it on shipped poses - and "
                       "the export is byte-identical either way")]
        + [RigProperty(tool_bind_prop(j), "bool", 1.0, 0.0, 1.0,
                       "%s follows the hand. Off, the socket holds its world pose while "
                       "the arm moves and the exported %s track counter-animates its "
                       "parent - key it on every frame, or it drifts between keys"
                       % (j, j)) for j in TOOL_JOINTS]
        + [RigProperty(tool_keyed_prop(j), "bool", 0.0, 0.0, 1.0,
                       "Treat %s's bind as an animation channel: the toggle keys it, and "
                       "keys the world-place hold with it. The switch is a hard cut, so "
                       "the channel is keyed CONSTANT" % j) for j in TOOL_JOINTS])
    for l in rig.limbs:
        end = "wrist" if l.key.startswith("arm") else "ankle"
        rig.properties.extend([
            RigProperty(l.end_lock, "float", 1.0, 0.0, 1.0,
                        "Under IK, how much %s's %s takes its roll from the IK control. "
                        "At 0 it keeps the roll the solve gave it, so the limb can be "
                        "planted and still turned by hand" % (l.label, end)),
            RigProperty(l.pole_follow, "bool", DEFAULT_POLE_FOLLOW, 0.0, 1.0,
                        "%s's pole rides the limb's own chord. On, %.2f%% of %d shipped "
                        "poses fold to the side Epic Fight's clips fold to; off, the pole "
                        "stands still in armature space the way the reference rig's does "
                        "and %.2f%% fold the wrong way"
                        % (l.label, POLE_FOLLOW_MEASURED["chord"],
                           POLE_FOLLOW_MEASURED["poses"], POLE_FOLLOW_MEASURED["world"])),
        ])

    rig.bones = [b if b.widget else replace(b, lock_select=True) for b in rig.bones]

    rig.widgets = {b.widget: WIDGETS[b.widget] for b in rig.bones if b.widget}
    return rig


@dataclass(frozen=True)
class _LimbSpec:
    key: str
    label: str
    upper: str
    lower: str
    marker: str
    ik_size: float
    fk_upper_size: float
    fk_lower_size: float
    limb: str


_ARM = (SIZES["ik_hand"], SIZES["fk_arm"], SIZES["fk_hand"], "Arm")
_LEG = (SIZES["ik_leg"], SIZES["fk_thigh"], SIZES["fk_leg"], "Leg")

_LIMBS = (
    _LimbSpec("arm_r", "Arm.R", "Arm_R", "Hand_R", "Elbow_R", *_ARM),
    _LimbSpec("arm_l", "Arm.L", "Arm_L", "Hand_L", "Elbow_L", *_ARM),
    _LimbSpec("leg_r", "Leg.R", "Thigh_R", "Leg_R", "Knee_R", *_LEG),
    _LimbSpec("leg_l", "Leg.L", "Thigh_L", "Leg_L", "Knee_L", *_LEG),
)


def _limb_collections(spec: _LimbSpec):
    side = spec.label.split(".")[1]
    return "FK %s.%s" % (spec.limb, side), "IK %s.%s" % (spec.limb, side)


def _build_limb(rig: RigDefinition, arm: Armature, rest, spec: _LimbSpec):
    """One limb, in the reference rig's shape.

    Three copies of the same two bones. The DEFORM pair is what the body is skinned to and
    what the exporter reads; it takes its control whole, first FK at influence 1 and then
    the solve at the blend. The FK pair is posed by hand. The IK pair is an exact duplicate
    of the deform rests carrying one IK constraint, chain_count 2, and it is the only place
    a solver is involved. Everything else in here is the four things we keep on top of that
    shape and each one is one bone: the chord, its roll reference, the pole's holder, and
    the point the end lock aims the lower segment at.
    """
    def add(bone):
        rig.bones.append(_place(bone))
    con = rig.constraints.append
    drive = rig.drivers.append

    up, low, mark = spec.upper, spec.lower, spec.marker
    fk_coll, ik_coll = _limb_collections(spec)
    prop = "ik_" + spec.key
    parent = arm[up].parent

    ik_ctrl_rest = _along(rest[low], LENGTHS[low])
    pole_rest = Mat4.translation(*pole_site(
        _head(rest[up]), _head(rest[low]), _head(ik_ctrl_rest),
        _bone_axis(rest[mark]), POLE_DISTANCE[mark])) @ _rot_only(rest[mark])
    angle = pole_angle(_head(rest[up]), _head(_along(rest[up], LENGTHS[up])),
                       _head(ik_ctrl_rest), _head(pole_rest), _x_axis(rest[up]))
    lo, hi = ik_band(spec.key)

    detach, twin = detach_ctrl(up), detach_twin(up)
    add(RigBone(detach, rest[up], 0.12, parent, "Detach",
                widget="wedge", widget_size=SIZES["detach"], role=ROTATOR))
    add(RigBone(twin, rest[up], 0.1, CTRL + "Master", "Mechanism", hide=True))
    con(RigConstraint(twin, DETACH_COPY, "COPY_TRANSFORMS", detach,
                      target_space="LOCAL", owner_space="LOCAL"))

    add(RigBone(FK + up, rest[up], LENGTHS[up], detach, fk_coll,
                widget="collar", widget_size=spec.fk_upper_size, lock_location=True,
                role=ROTATOR))
    add(RigBone(FK + low, rest[low], LENGTHS[low], FK + up, fk_coll,
                widget="collar", widget_size=spec.fk_lower_size, lock_location=True,
                role=ROTATOR))

    seed = IK_SEED[spec.key] * FOLD_SIGN[spec.key]
    seed_up, seed_low = seed_chain(rest[up], LENGTHS[up], rest[low], seed)
    add(RigBone(MCH + up + ".ik", seed_up, LENGTHS[up], detach, "Mechanism", hide=True))
    add(RigBone(MCH + low + ".ik", seed_low, LENGTHS[low], MCH + up + ".ik",
                "Mechanism", hide=True, ik_lock=HINGE_LOCK,
                ik_limit=(lo - seed, hi - seed)))

    add(RigBone(MCH + up + ".twist", seed_up, LENGTHS[up], MCH + up + ".ik",
                "Mechanism", hide=True))

    add(RigBone(IK + low, ik_ctrl_rest, LENGTHS[low] * 0.5, twin, ik_coll,
                widget="cube", widget_size=spec.ik_size, role=HANDLE))

    chord_dir = normalize(sub(_head(ik_ctrl_rest), _head(rest[up])))
    ref_dir = normalize(_from_frame(rest[parent], POLE_REFERENCE[spec.key]))
    ref_head = tuple(h + d * POLE_REF_ARM for h, d in zip(_head(rest[up]), ref_dir))
    add(RigBone(MCH + POLE + mark + ".ref",
                Mat4.translation(*ref_head) @ _frame(ref_dir, chord_dir),
                0.1, detach, "Mechanism", hide=True))
    add(RigBone(MCH + POLE + mark + ".chord",
                Mat4.translation(*_head(rest[up])) @ _frame(chord_dir, ref_dir),
                0.2, detach, "Mechanism", hide=True))
    con(RigConstraint(MCH + POLE + mark + ".chord", "Chord", "DAMPED_TRACK", IK + low,
                      track_axis="TRACK_Y"))
    con(RigConstraint(MCH + POLE + mark + ".chord", "Roll", "LOCKED_TRACK",
                      MCH + POLE + mark + ".ref", track_axis="TRACK_X", lock_axis="LOCK_Y"))

    add(RigBone(MCH + POLE + mark + ".space", pole_rest, 0.12, None, "Mechanism",
                hide=True))
    con(RigConstraint(MCH + POLE + mark + ".space", POLE_FOLLOW_SPACE, "CHILD_OF",
                      MCH + POLE + mark + ".chord", influence=DEFAULT_POLE_FOLLOW,
                      seat_inverse=True))
    drive(RigDriver(MCH + POLE + mark + ".space", POLE_FOLLOW_SPACE, "f",
                    (DriverVar("f", pole_follow_prop(spec.key)),)))
    con(RigConstraint(MCH + POLE + mark + ".space", POLE_ROOT_SPACE, "CHILD_OF",
                      twin, influence=1.0 - DEFAULT_POLE_FOLLOW,
                      seat_inverse=True))
    drive(RigDriver(MCH + POLE + mark + ".space", POLE_ROOT_SPACE, "1 - f",
                    (DriverVar("f", pole_follow_prop(spec.key)),)))
    add(RigBone(POLE + mark, pole_rest, 0.12, MCH + POLE + mark + ".space", ik_coll,
                widget="ball", widget_size=SIZES["pole"], role=HANDLE))

    add(RigBone(MCH + IK + low + ".roll", _sideways(ik_ctrl_rest, ROLL_ARM), 0.06,
                IK + low, "Mechanism", hide=True))

    con(RigConstraint(MCH + low + ".ik", "IK", "IK", IK + low,
                      chain_count=2, pole_subtarget=POLE + mark, pole_angle=angle,
                      use_rotation=False, use_stretch=False))

    for joint, fk_bone, ik_bone in ((up, FK + up, MCH + up + ".twist"),
                                    (low, FK + low, MCH + low + ".ik")):
        con(RigConstraint(joint, "FK", "COPY_TRANSFORMS", fk_bone))
        con(RigConstraint(joint, "IK", "COPY_TRANSFORMS", ik_bone, influence=0.0))
        drive(RigDriver(joint, "IK", "ik", (DriverVar("ik", prop),)))

    con(RigConstraint(low, "Roll", "LOCKED_TRACK", MCH + IK + low + ".roll",
                      influence=0.0, track_axis="TRACK_X", lock_axis="LOCK_Y"))
    drive(RigDriver(low, "Roll", "ik * lock",
                    (DriverVar("ik", prop), DriverVar("lock", end_lock_prop(spec.key)))))


    vis = rig.collection_drivers.append
    show = DriverVar("show", SHOW_INACTIVE)
    vis(RigCollectionDriver(fk_coll, "max(show, ceil(1.0 - ik))",
                            (DriverVar("ik", prop), show)))
    vis(RigCollectionDriver(ik_coll, "max(show, ceil(ik))",
                            (DriverVar("ik", prop), show)))

    rig.limbs.append(IkLimb(
        key=spec.key, label=spec.label, prop=prop, upper=up, lower=low, marker=mark,
        fk_upper=FK + up, fk_lower=FK + low,
        ik_upper=MCH + up + ".ik", ik_lower=MCH + low + ".ik",
        ik_twist=MCH + up + ".twist",
        ik_ctrl=IK + low, pole_ctrl=POLE + mark,
        fk_collection=fk_coll, ik_collection=ik_coll, pole_angle=angle,
        default_ik=DEFAULT_IK[spec.key],
        end_lock=end_lock_prop(spec.key), end_roll=MCH + IK + low + ".roll",
        span=LENGTHS[up] + LENGTHS[low],
        pole_space=MCH + POLE + mark + ".space",
        pole_chord=MCH + POLE + mark + ".chord", pole_ref=MCH + POLE + mark + ".ref",
        pole_follow=pole_follow_prop(spec.key), ik_seed=seed,
        detach=detach, detach_twin=twin))




def limbs_to_json(limbs) -> str:
    keys = [f.name for f in fields(IkLimb)]
    return json.dumps([{k: getattr(l, k) for k in keys} for l in limbs])


def limbs_from_json(text) -> list:
    tuples = {f.name for f in fields(IkLimb) if f.type == "tuple"}
    keys = {f.name for f in fields(IkLimb)}
    return [IkLimb(**{k: (tuple(v) if k in tuples else v)
                      for k, v in d.items() if k in keys})
            for d in json.loads(text)]




_CONSTRAINT_TYPES = {"COPY_TRANSFORMS", "COPY_LOCATION", "COPY_SCALE", "COPY_ROTATION",
                     "IK", "DAMPED_TRACK", "LOCKED_TRACK", "CHILD_OF", "LIMIT_DISTANCE"}

_SPACES = {"WORLD", "LOCAL", "POSE", "LOCAL_WITH_PARENT", "CUSTOM", "OWNER_LOCAL"}

_LIMIT_MODES = {"LIMITDIST_INSIDE", "LIMITDIST_OUTSIDE", "LIMITDIST_ONSURFACE"}

_FLAG_PATHS = {"ik_stretch": (False, -1), "use_inherit_rotation": (True, -1),
               "lock_rotation": (False, 2), "hide": (True, -1)}


def validate(rig: RigDefinition) -> list:
    """Every structural claim the bpy layer relies on. Returns a list of errors."""
    errors = []
    names = [b.name for b in rig.bones]
    index = {}
    for b in rig.bones:
        if b.name in index:
            errors.append("duplicate bone %r" % b.name)
        index[b.name] = b

    collections = {c.name for c in rig.collections}
    props = {p.name for p in rig.properties}
    prop_default = {p.name: p.default for p in rig.properties}

    seen = set()
    for b in rig.bones:
        if b.parent is not None:
            if b.parent not in index:
                errors.append("bone %r has unknown parent %r" % (b.name, b.parent))
            elif b.parent not in seen:
                errors.append("bone %r is built before its parent %r"
                              % (b.name, b.parent))
        seen.add(b.name)
        if b.collection not in collections:
            errors.append("bone %r is in undeclared collection %r"
                          % (b.name, b.collection))
        if b.widget and b.widget not in WIDGETS:
            errors.append("bone %r wants unknown widget %r" % (b.name, b.widget))
        if b.length <= 0.0:
            errors.append("bone %r has non-positive length" % b.name)
        for attr in ("widget_translation", "widget_rotation"):
            v = getattr(b, attr)
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                errors.append("bone %r has a bad %s %r" % (b.name, attr, v))
            elif not b.widget and tuple(v) != _ZERO:
                errors.append("bone %r has a %s but no widget to place" % (b.name, attr))
        if bool(b.widget) != (b.role in (ROTATOR, HANDLE)):
            errors.append("bone %r is drawn as %r but its role is %r"
                          % (b.name, b.widget, b.role))
        if b.lock_select == bool(b.widget):
            errors.append("bone %r is %s and %s selectable"
                          % (b.name, "drawn" if b.widget else "undrawn",
                             "not" if b.lock_select else ""))
        if (b.role == ROTATOR and tuple(b.widget_translation) != _ZERO
                and b.name not in PIVOT_EXCEPTIONS):
            errors.append("rotator %r draws its widget off its pivot with no entry in "
                          "PIVOT_EXCEPTIONS" % b.name)

    seen_props = set()
    for p in rig.properties:
        if p.name in seen_props:
            errors.append("property %r is declared twice" % p.name)
        seen_props.add(p.name)
        if p.kind not in ("float", "bool", "int"):
            errors.append("property %r has unsupported kind %r" % (p.name, p.kind))
        if not p.minimum <= p.default <= p.maximum:
            errors.append("property %r defaults outside its range" % p.name)
        if p.kind == "int" and len(p.items) != int(p.maximum) - int(p.minimum) + 1:
            errors.append("property %r has %d labels for %d values"
                          % (p.name, len(p.items), int(p.maximum - p.minimum) + 1))
        if p.kind != "int" and p.items:
            errors.append("property %r is not an enum but carries labels" % p.name)

    for c in rig.collections:
        theme = COLOURS.get(c.name)
        if theme is None:
            errors.append("collection %r has no theme colour" % c.name)
        elif theme == "DEFAULT" and c.name != UNCOLOURED:
            errors.append("collection %r is left on the default grey" % c.name)
        if c.name.endswith(".R") and COLOURS.get(c.name[:-2] + ".L") == theme:
            errors.append("collection %r wears the same theme as its left twin" % c.name)

    for name in sorted(set(WIDGET_OFFSETS) - set(index)):
        errors.append("widget offset names unknown bone %r" % name)
    for name in sorted(PIVOT_EXCEPTIONS):
        if index.get(name) is None or index[name].role != ROTATOR:
            errors.append("pivot exception %r is not a rotator" % name)
        elif not PIVOT_EXCEPTIONS[name].strip():
            errors.append("pivot exception %r gives no reason" % name)

    for b in rig.bones:
        cur, hops = b.parent, 0
        while cur is not None and hops <= len(names):
            if cur == b.name:
                errors.append("parent cycle through %r" % b.name)
                break
            cur = index[cur].parent if cur in index else None
            hops += 1

    per_owner = {}
    for c in rig.constraints:
        if c.type not in _CONSTRAINT_TYPES:
            errors.append("constraint %r on %r has unsupported type %r"
                          % (c.name, c.owner, c.type))
        if c.owner not in index:
            errors.append("constraint %r targets unknown owner %r" % (c.name, c.owner))
            continue
        if c.name in per_owner.setdefault(c.owner, set()):
            errors.append("bone %r has two constraints named %r" % (c.owner, c.name))
        per_owner[c.owner].add(c.name)
        if not c.subtarget:
            errors.append("constraint %r on %r has no subtarget" % (c.name, c.owner))
        elif c.subtarget not in index:
            errors.append("constraint %r on %r targets unknown bone %r"
                          % (c.name, c.owner, c.subtarget))
        if c.type == "IK":
            if c.chain_count < 1:
                errors.append("IK on %r has no chain length" % c.owner)
            cur = c.owner
            for _ in range(c.chain_count - 1):
                cur = index[cur].parent if cur in index else None
                if cur is None:
                    errors.append("IK chain on %r runs past the armature root" % c.owner)
                    break
            if c.pole_subtarget and c.pole_subtarget not in index:
                errors.append("IK on %r has unknown pole %r" % (c.owner, c.pole_subtarget))
        if c.type == "LIMIT_DISTANCE":
            if c.limit_mode not in _LIMIT_MODES:
                errors.append("limit distance on %r has mode %r" % (c.owner, c.limit_mode))
            if not c.distance > 0.0:
                errors.append("limit distance on %r has no distance" % c.owner)
        if c.type in ("DAMPED_TRACK", "LOCKED_TRACK") and not c.track_axis:
            errors.append("%s on %r has no track axis" % (c.type, c.owner))
        if c.type == "LOCKED_TRACK":
            if not c.lock_axis:
                errors.append("locked track on %r has no lock axis" % c.owner)
            elif c.lock_axis[-1] == c.track_axis[-1]:
                errors.append("locked track on %r tracks the axis it locks" % c.owner)
        if c.type == "COPY_ROTATION" and len(c.use_axis) != 3:
            errors.append("copy rotation on %r names no axes" % c.owner)
        for space in (c.target_space, c.owner_space):
            if space and space not in _SPACES:
                errors.append("constraint %r on %r asks for space %r"
                              % (c.name, c.owner, space))
        if (c.target_space or c.owner_space) and c.type not in ("COPY_ROTATION",
                                                                "COPY_TRANSFORMS"):
            errors.append("constraint %r on %r sets a space the builder ignores"
                          % (c.name, c.owner))

    solved = {c.owner for c in rig.constraints if c.type == "IK"}
    for b in rig.bones:
        if not b.ik_lock and not b.ik_limit:
            continue
        if b.name not in solved:
            errors.append("bone %r locks IK axes but no IK constraint bends it" % b.name)
        if tuple(b.ik_lock) != HINGE_LOCK:
            errors.append("bone %r is not a one-axis hinge: locks %r"
                          % (b.name, b.ik_lock))
        if len(b.ik_limit) != 2 or not all(math.isfinite(v) for v in b.ik_limit):
            errors.append("bone %r has a bad IK limit %r" % (b.name, b.ik_limit))
        elif b.ik_limit[0] >= b.ik_limit[1]:
            errors.append("bone %r has an empty IK limit %r" % (b.name, b.ik_limit))
        elif not b.ik_limit[0] <= 0.0 <= b.ik_limit[1]:
            errors.append("bone %r cannot reach its own rest inside its IK limit %r"
                          % (b.name, b.ik_limit))

    deps = {n: set() for n in index}
    for b in rig.bones:
        if b.parent in index:
            deps[b.name].add(b.parent)
    for c in rig.constraints:
        if c.owner not in index:
            continue
        chain = [c.owner]
        if c.type == "IK":
            cur = c.owner
            for _ in range(c.chain_count - 1):
                cur = index[cur].parent
                if cur is None:
                    break
                chain.append(cur)
        for member in chain:
            for target in (c.subtarget, c.pole_subtarget):
                if target and target in index and target != member:
                    deps[member].add(target)
    errors.extend(_cycles(deps))

    def check_vars(where, variables, expression):
        for v in variables:
            if isinstance(v, DistanceVar):
                for b in (v.bone_a, v.bone_b):
                    if b not in index:
                        errors.append("driver on %r measures to unknown bone %r"
                                      % (where, b))
            elif v.prop not in props:
                errors.append("driver on %r reads undeclared property %r"
                              % (where, v.prop))
            if v.name not in expression:
                errors.append("driver on %r never uses variable %r" % (where, v.name))

    for d in rig.drivers:
        if d.bone not in index:
            errors.append("driver on unknown bone %r" % d.bone)
        elif d.constraint not in per_owner.get(d.bone, set()):
            errors.append("driver targets missing constraint %r on %r"
                          % (d.constraint, d.bone))
        check_vars(d.bone, d.variables, d.expression)

    seen_flags = set()
    for d in rig.flag_drivers:
        if d.bone not in index:
            errors.append("flag driver on unknown bone %r" % d.bone)
        spec = _FLAG_PATHS.get(d.path)
        if spec is None:
            errors.append("flag driver on %r writes unsupported path %r"
                          % (d.bone, d.path))
        else:
            on_data, top = spec
            if d.on_data != on_data:
                errors.append("flag driver %r on %r is on the wrong datablock"
                              % (d.path, d.bone))
            if top < 0 and d.index >= 0:
                errors.append("flag driver %r on %r indexes a scalar" % (d.path, d.bone))
            if top >= 0 and not 0 <= d.index <= top:
                errors.append("flag driver %r on %r has index %d out of range"
                              % (d.path, d.bone, d.index))
        if (d.bone, d.path, d.index) in seen_flags:
            errors.append("two flag drivers write %r on %r" % (d.path, d.bone))
        seen_flags.add((d.bone, d.path, d.index))
        check_vars(d.bone, d.variables, d.expression)

    driven = set()
    for d in rig.collection_drivers:
        if d.collection not in collections:
            errors.append("visibility driver on undeclared collection %r" % d.collection)
        if d.collection in driven:
            errors.append("collection %r has two visibility drivers" % d.collection)
        driven.add(d.collection)
        check_vars(d.collection, d.variables, d.expression)
    for l in rig.limbs:
        for coll in (l.fk_collection, l.ik_collection):
            if coll not in driven:
                errors.append("limb %s collection %r is not driven by its blend"
                              % (l.key, coll))

    arm = rig.armature
    deform = {b.ef_joint: b for b in rig.bones if b.ef_joint}
    if set(deform) != set(arm.names()):
        errors.append("deform bones are not the Epic Fight joint set: %s"
                      % sorted(set(deform) ^ set(arm.names())))
    for jname, b in deform.items():
        if b.name != jname:
            errors.append("deform bone %r renames Epic Fight joint %r" % (b.name, jname))
        if b.parent != arm[jname].parent:
            errors.append("deform bone %r reparents %r -> %r"
                          % (jname, arm[jname].parent, b.parent))
        if not b.rest.approx_equal(arm.rest_model(jname), 1e-12):
            errors.append("deform bone %r rest differs from biped.json" % jname)
        if not b.use_deform:
            errors.append("deform bone %r is not marked deforming" % jname)
    for c in rig.constraints:
        if c.owner in deform and c.subtarget in deform:
            errors.append("deform bone %r is driven by another deform bone %r"
                          % (c.owner, c.subtarget))

    for l in rig.limbs:
        for attr in ("fk_upper", "fk_lower", "ik_upper", "ik_lower", "ik_ctrl",
                     "pole_ctrl", "upper", "lower", "marker",
                     "pole_space", "pole_chord", "pole_ref", "end_roll",
                     "detach", "detach_twin"):
            b = getattr(l, attr)
            if b not in index:
                errors.append("limb %s.%s names unknown bone %r" % (l.key, attr, b))
        for attr in ("prop", "end_lock", "pole_follow"):
            if getattr(l, attr) not in props:
                errors.append("limb %s uses undeclared property %r"
                              % (l.key, getattr(l, attr)))
        deform_up = index.get(l.upper)
        det, twin = index.get(l.detach), index.get(l.detach_twin)
        if det is not None and deform_up is not None:
            if det.parent != deform_up.parent:
                errors.append("limb %s hangs its detach off %r, not the limb's own "
                              "socket %r" % (l.key, det.parent, deform_up.parent))
            if not det.rest.approx_equal(deform_up.rest, 1e-12):
                errors.append("limb %s detach does not rest on the limb root" % l.key)
        if twin is not None:
            if twin.parent != CTRL + "Master":
                errors.append("limb %s parents its detach twin to %r, not to the master"
                              % (l.key, twin.parent))
            if det is not None and not twin.rest.approx_equal(det.rest, 1e-12):
                errors.append("limb %s detach twin does not rest on its control" % l.key)
            stack = [c for c in rig.constraints if c.owner == l.detach_twin]
            want_twin = [(DETACH_COPY, "COPY_TRANSFORMS", l.detach, "LOCAL", "LOCAL")]
            if [(c.name, c.type, c.subtarget, c.target_space, c.owner_space)
                    for c in stack] != want_twin:
                errors.append("limb %s detach twin carries %s, not one LOCAL copy of its "
                              "control" % (l.key, [(c.name, c.subtarget) for c in stack]))
        handle = index.get(l.ik_ctrl)
        if handle is not None and handle.parent != l.detach_twin:
            errors.append("limb %s parents its IK handle to %r, not to the detach twin"
                          % (l.key, handle.parent))
        if [c for c in rig.constraints if c.owner == l.ik_ctrl]:
            errors.append("limb %s puts a constraint on its IK handle, so a snap writing "
                          "pose_bone.matrix would land displaced" % l.key)
        pole_space = index.get(l.pole_space)
        if pole_space is not None:
            if pole_space.parent is not None:
                errors.append("limb %s pole holder is parented, so its two Child Ofs would "
                              "each apply the root a second time" % l.key)
            if not pole_space.rest.approx_equal(index[l.pole_ctrl].rest, 1e-12):
                errors.append("limb %s pole holder does not rest on its pole" % l.key)
        if index.get(l.pole_ctrl) is not None and index[l.pole_ctrl].parent != l.pole_space:
            errors.append("limb %s parents its pole to %r, not to its chord holder"
                          % (l.key, index[l.pole_ctrl].parent))
        stack = [c for c in rig.constraints
                 if c.owner == l.pole_space and c.type == "CHILD_OF"]
        want = [(POLE_FOLLOW_SPACE, l.pole_chord), (POLE_ROOT_SPACE, l.detach_twin)]
        if [(c.name, c.subtarget) for c in stack] != want:
            errors.append("limb %s pole holder carries %s, not the chord and the detach "
                          "twin" % (l.key, [(c.name, c.subtarget) for c in stack]))
        drivers = {d.constraint: d.expression for d in rig.drivers
                   if d.bone == l.pole_space}
        if drivers != {POLE_FOLLOW_SPACE: "f", POLE_ROOT_SPACE: "1 - f"}:
            errors.append("limb %s pole spaces are driven %s, not complementary"
                          % (l.key, drivers))
        solved_set = {l.upper, l.lower, l.marker, l.ik_upper, l.ik_lower,
                      l.pole_ctrl, l.pole_space}
        for c in rig.constraints:
            if c.owner == l.pole_chord and c.subtarget in solved_set:
                errors.append("limb %s chord helper reads %r, which the solve produces"
                              % (l.key, c.subtarget))
        for attr in ("pole_chord", "pole_ref"):
            helper = index.get(getattr(l, attr))
            if helper is not None and helper.parent != l.detach:
                errors.append("limb %s hangs %s off %r, not the limb's own detach %r"
                              % (l.key, attr, helper.parent, l.detach))
        for chain in (l.fk_upper, l.ik_upper):
            if index.get(chain) is not None and index[chain].parent != l.detach:
                errors.append("limb %s hangs %r off %r, not the limb's own detach %r"
                              % (l.key, chain, index[chain].parent, l.detach))
        for attr, joint in (("fk_upper", l.upper), ("fk_lower", l.lower)):
            copy = index.get(getattr(l, attr))
            if copy is not None and index.get(joint) is not None:
                if not copy.rest.approx_equal(index[joint].rest, 1e-12):
                    errors.append("limb %s %s does not rest on %r" % (l.key, attr, joint))
        seed = abs(IK_SEED.get(l.key, 0.0))
        for attr, joint in (("ik_upper", l.upper), ("ik_lower", l.lower)):
            copy, joint_bone = index.get(getattr(l, attr)), index.get(joint)
            if copy is None or joint_bone is None:
                continue
            span = 2.0 * math.sin(math.radians(seed) * 0.5) * (l.span or 1.0)
            if length(sub(_head(copy.rest), _head(joint_bone.rest))) > span + 1e-9:
                errors.append("limb %s %s is more than IK_SEED off %r" % (l.key, attr, joint))
        if seed > max(abs(v) for v in ik_band(l.key)):
            errors.append("limb %s seeds the solver outside its own band" % l.key)
        if index.get(l.ik_lower) is not None and not index[l.ik_lower].ik_lock:
            errors.append("limb %s solves a free joint at %r, not a hinge"
                          % (l.key, l.ik_lower))
        roll = index.get(l.end_roll)
        if roll is not None and roll.parent != l.ik_ctrl:
            errors.append("limb %s aims its end lock at %r, which does not ride the handle"
                          % (l.key, roll.parent))
        if not l.span > 0.0:
            errors.append("limb %s has no span" % l.key)
        elif prop_default.get(l.prop) != l.default_ik:
            errors.append("limb %s starts at %r but its property defaults to %r"
                          % (l.key, l.default_ik, prop_default.get(l.prop)))
        if not 0.0 <= l.default_ik <= 1.0:
            errors.append("limb %s has an out of range default %r" % (l.key, l.default_ik))
        for coll in (l.fk_collection, l.ik_collection):
            if coll not in collections:
                errors.append("limb %s names undeclared collection %r" % (l.key, coll))
            elif _visible(rig, coll) != (l.default_ik < 1.0 if coll == l.fk_collection
                                         else l.default_ik > 0.0):
                errors.append("limb %s collection %r starts out of step with the blend"
                              % (l.key, coll))
        if not math.isfinite(l.pole_angle):
            errors.append("limb %s has a non-finite pole angle" % l.key)
    return errors


def _visible(rig: RigDefinition, name) -> bool:
    for c in rig.collections:
        if c.name == name:
            return c.visible
    return False


def _cycles(deps) -> list:
    """Iterative DFS over the dependency graph; one message per cycle found."""
    WHITE, GREY, BLACK = 0, 1, 2
    state = {n: WHITE for n in deps}
    out = []
    for start in deps:
        if state[start] != WHITE:
            continue
        stack = [(start, iter(sorted(deps[start])))]
        state[start] = GREY
        path = [start]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[node] = BLACK
                stack.pop()
                path.pop()
                continue
            if state.get(nxt) == GREY:
                out.append("pose dependency cycle: %s"
                           % " -> ".join(path[path.index(nxt):] + [nxt]))
            elif state.get(nxt) == WHITE:
                state[nxt] = GREY
                path.append(nxt)
                stack.append((nxt, iter(sorted(deps[nxt]))))
    return out
