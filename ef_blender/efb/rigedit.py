"""Tables the edit panel edits. No bpy, so the numbers can be tested and argued about
without a Blender.

Everything here is a default. The live values sit in a PropertyGroup on the rig object
(efbpy.rigedit), which is what survives a save and what two rigs in one file keep apart.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WidgetGroup", "WIDGET_GROUPS", "group_of", "Hinge", "HINGE", "hinges",
           "COLLECTION_LABELS", "TOOL_COLLECTION", "HINGE_CONSTRAINT",
           "AxisLimit", "AXIS_LIMITS",
           "AXIS_GROUPS", "PRESETS", "EF_PRESET", "HUMAN_PRESET", "axis_ranges",
           "axis_limits_for", "LIMIT_CONSTRAINT", "Seam", "SEAMS", "seams",
           "SEAM_CONSTRAINT", "seam_slide", "TOOL_SOCKETS", "PIN_AXES",
           "PIN_CONSTRAINT", "DEFORM_PIN", "DEFORM_ROLL_PIN", "ROLL_PIN_ORDER",
           "REPLAY_PROP", "PIN_EPSILON", "TOOL_AXIS", "TOOL_AXIS_SIGN",
           "TOOL_WIDGET_TURN"]

HINGE_CONSTRAINT = "Hinge"
LIMIT_CONSTRAINT = "Axis Limits"
SEAM_CONSTRAINT = "Seam"
PIN_CONSTRAINT = "Off-axis pin"
DEFORM_PIN = "Off-axis pin (export)"
DEFORM_ROLL_PIN = "Off-axis roll (export)"
ROLL_PIN_ORDER = "YXZ"
TOOL_COLLECTION = "Sockets"

TOOL_SOCKETS = ("Tool_R", "Tool_L", "Chest")

TOOL_AXIS = 2
TOOL_AXIS_SIGN = -1.0

_TURN_ONTO_AXIS = {(0, 1.0): (0.0, 90.0, 0.0), (0, -1.0): (0.0, -90.0, 0.0),
                   (1, 1.0): (-90.0, 0.0, 0.0), (1, -1.0): (90.0, 0.0, 0.0),
                   (2, 1.0): (0.0, 0.0, 0.0), (2, -1.0): (0.0, 180.0, 0.0)}
TOOL_WIDGET_TURN = _TURN_ONTO_AXIS[(TOOL_AXIS, TOOL_AXIS_SIGN)]


@dataclass(frozen=True)
class WidgetGroup:
    key: str
    label: str
    icon: str
    prefixes: tuple
    exact: tuple = ()

    def holds(self, bone: str) -> bool:
        return bone in self.exact or any(bone.startswith(p) for p in self.prefixes)


WIDGET_GROUPS = (
    WidgetGroup("root", "Root", "OUTLINER_OB_EMPTY", (),
                ("CTRL-Master", "CTRL-COG", "CTRL-Root")),
    WidgetGroup("spine", "Spine and head", "BONE_DATA", ("CTRL-Shoulder",),
                ("CTRL-Torso", "CTRL-Chest", "CTRL-Head")),
    WidgetGroup("detach", "Limb detach", "UNLINKED", ("CTRL-Detach-",)),
    WidgetGroup("fk", "FK limbs", "CON_ROTLIKE", ("FK-",)),
    WidgetGroup("ik", "IK handles", "CON_KINEMATIC", ("IK-",)),
    WidgetGroup("pole", "Poles", "CON_TRACKTO", ("POLE-",)),
    WidgetGroup("tool", "Tool sockets", "TOOL_SETTINGS", ("CTRL-Tool",)),
)


def group_of(bone: str):
    for g in WIDGET_GROUPS:
        if g.holds(bone):
            return g
    return None


@dataclass(frozen=True)
class Hinge:
    limb: str
    bone: str
    ik_bone: str
    deform: str
    axis: int
    minimum: float
    maximum: float
    enabled: bool
    slack: tuple = ()
    seed: float = 0.0


HINGE = {
    "arm": (0, -35.0, 140.0, True, (None, None)),
    "leg": (0, -160.0, 5.0, True, ((-15.0, 15.0), (-20.0, 15.0))),
}

PIN_AXES = (1, 2)

REPLAY_PROP = "clip_replay"

PIN_EPSILON = 0.5


def hinges(limbs) -> list:
    """One Hinge per IkLimb, keyed off the limb's own bone names."""
    out = []
    for limb in limbs:
        axis, lo, hi, on, slack = HINGE["arm" if limb.key.startswith("arm") else "leg"]
        out.append(Hinge(limb.key, limb.fk_lower, limb.ik_lower, limb.lower,
                         axis, lo, hi, on, slack, limb.ik_seed))
    return out


@dataclass(frozen=True)
class AxisLimit:
    """A rotation box on one control that is not a hinge."""
    bone: str
    group: str
    order: str
    envelope: tuple
    human: tuple
    note: str = ""


AXIS_GROUPS = {
    "spine": ("Spine and neck", "BONE_DATA", True),
    "clavicle": ("Collarbones", "CON_TRACKTO", True),
    "ball": ("Shoulders and hips", "CON_ROTLIKE", False),
}

EF_PRESET = "EF"
HUMAN_PRESET = "HUMAN"
PRESETS = (EF_PRESET, HUMAN_PRESET)

AXIS_LIMITS = (
    AxisLimit("CTRL-Torso", "spine", "XZY",
              ((-135.0, 65.0), (-40.0, 50.0), (-30.0, 30.0)),
              ((-90.0, 35.0), (-40.0, 40.0), (-30.0, 30.0)),
              "measured -109.6/+41.1, -26.3/+38.8, -19.6/+16.4; gimbal 19.6"),
    AxisLimit("CTRL-Chest", "spine", "XZY",
              ((-80.0, 60.0), (-65.0, 65.0), (-40.0, 40.0)),
              ((-50.0, 25.0), (-40.0, 40.0), (-30.0, 30.0)),
              "measured -60.4/+43.8, -48.5/+49.7, -29.1/+28.9; gimbal 29.1"),
    AxisLimit("CTRL-Head", "spine", "XZY",
              ((-90.0, 115.0), (-145.0, 145.0), (-40.0, 40.0)),
              ((-50.0, 60.0), (-80.0, 80.0), (-45.0, 45.0)),
              "measured -65.6/+89.7, -106.9/+110.9, -27.7/+25.0; gimbal 27.7"),
    AxisLimit("CTRL-Shoulder_R", "clavicle", "ZXY",
              ((-45.0, 65.0), (-115.0, 115.0), (-120.0, 120.0)),
              ((-25.0, 25.0), (-30.0, 30.0), (-25.0, 25.0)),
              "measured -30.2/+50.5, -87.4/+72.2, -93.1/+82.6; gimbal 50.5"),
    AxisLimit("CTRL-Shoulder_L", "clavicle", "ZXY",
              ((-45.0, 65.0), (-115.0, 115.0), (-120.0, 120.0)),
              ((-25.0, 25.0), (-30.0, 30.0), (-25.0, 25.0)),
              "mirror of CTRL-Shoulder_R"),
    AxisLimit("FK-Arm_R", "ball", "ZXY", (None, None, None), (None, None, None),
              "free: gimbal 86.8 in the best of six orders"),
    AxisLimit("FK-Arm_L", "ball", "ZXY", (None, None, None), (None, None, None),
              "free: gimbal 87.5 in the best of six orders"),
    AxisLimit("FK-Thigh_R", "ball", "XZY", (None, None, None),
              ((-30.0, 120.0), (-45.0, 45.0), (-45.0, 45.0)),
              "free: shipped clips use -128.5/+147.7 on the fold"),
    AxisLimit("FK-Thigh_L", "ball", "XZY", (None, None, None),
              ((-30.0, 120.0), (-45.0, 45.0), (-45.0, 45.0)),
              "mirror of FK-Thigh_R"),
)


def axis_ranges(row: AxisLimit, preset: str) -> tuple:
    return row.human if preset == HUMAN_PRESET else row.envelope


def axis_limits_for(group: str) -> list:
    return [r for r in AXIS_LIMITS if r.group == group]


@dataclass(frozen=True)
class Seam:
    """The slide that keeps a two-box limb's hairline shut through a fold.

    Epic Fight's arm and leg are two boxes sharing a double ring of vertices 0.0005 apart.
    Folding rotates the rings apart; sliding the seam joint along its own local Z pulls
    its ring back into the other one.
    """
    marker: str
    lower: str
    from_x: tuple
    to_z: tuple


SEAMS = (
    Seam("Elbow_R", "Hand_R", (0.0, 90.0), (0.0, -0.1472)),
    Seam("Elbow_L", "Hand_L", (0.0, 90.0), (0.0, -0.1472)),
    Seam("Knee_R", "Leg_R", (-90.0, -30.0), (-0.11, 0.0)),
    Seam("Knee_L", "Leg_L", (-90.0, -30.0), (-0.11, 0.0)),
)


def seams(names) -> list:
    """The seams whose two bones both exist on this rig."""
    have = set(names)
    return [s for s in SEAMS if s.marker in have and s.lower in have]


def seam_slide(seam: Seam, degrees: float) -> float:
    """Local Z the marker takes at this fold. Blender's Transformation constraint clamps
    outside the mapped span rather than extrapolating, and so does this."""
    lo, hi = seam.from_x
    t = min(1.0, max(0.0, (degrees - lo) / (hi - lo)))
    return seam.to_z[0] + t * (seam.to_z[1] - seam.to_z[0])


COLLECTION_LABELS = {
    "Detach": ("Limb detach", "UNLINKED"),
    "Sockets": ("Tool controls", "TOOL_SETTINGS"),
    "Markers": ("Elbow and knee markers", "EMPTY_AXIS"),
    "Spine": ("Spine and head", "BONE_DATA"),
    "Root": ("Root and COG", "OUTLINER_OB_EMPTY"),
    "Deform": ("Deform bones", "GROUP_BONE"),
    "Mechanism": ("Mechanism", "SETTINGS"),
}
