"""Telling what version of this rig a saved file is. No bpy.

The stamp cannot answer it. `dist/ef_blender-0.8.0.zip` builds a rig stamping RIG_VERSION
6 while the tree at manifest 0.8.0 stamps 7; version 1 and 2 are both 70 bones; 6 and 7
have identical bone sets and are separated only by the property set. So the structure
answers instead, and the stamp is kept as a cross-check.
"""

from __future__ import annotations

__all__ = ["VERSION_MARKS", "DEFORM_ONLY", "ladder_version", "mark_for", "has_mark"]

VERSION_MARKS = (
    (12, "bone", "MCH-Detach-Arm_R"),
    (11, "parent", ("IK-Hand_R", "CTRL-Master")),
    (10, "prop", "pole_follow_arm_r"),
    (9, "bone", "MCH-Hand_R.twistfk"),
    (8, "bone", "MCH-Arm_R.twistfk"),
    (7, "prop", "joint_ctrl_arm_r"),
    (6, "bone", "CTRL-Detach-Arm_R"),
    (5, "bone", "MCH-POLE-Elbow_R.chord"),
    (4, "prop", "fold_flip_arm_r"),
    (3, "bone", "MCH-IK-Hand_R.goal"),
    (2, "prop", "show_inactive"),
    (1, "bone", "CTRL-Master"),
)

DEFORM_ONLY = 0


def has_mark(kind, mark, bones, props, parents) -> bool:
    """Is one VERSION_MARKS entry on a rig with these bones, properties and bone parents?"""
    if kind == "parent":
        return parents.get(mark[0]) == mark[1]
    return mark in (bones if kind == "bone" else props)


def ladder_version(bones, props, parents=None):
    """The newest first-appearance mark present, or None for a rig with no control.

    `parents` is {bone: parent name or None}. Left out, a parenting mark can never match
    and detection falls through to the newest mark that a name alone can carry.

    A name mark is taken at face value, a parenting is not. Version 2 already hung the IK
    handles off the master, so the mark for 11 is on a rig eight versions older too; it
    only names 11 when a name mark puts the rig at 10 already.
    """
    bones, props = set(bones), set(props)
    parents = parents or {}
    hit = [(v, kind) for v, kind, mark in VERSION_MARKS
           if has_mark(kind, mark, bones, props, parents)]
    named = max((v for v, kind in hit if kind != "parent"), default=None)
    if named is None:
        return None
    return max([named] + [v for v, kind in hit if kind == "parent" and named >= v - 1])


def mark_for(version) -> str:
    mark = next((m for v, _, m in VERSION_MARKS if v == version), "")
    return "%s under %s" % mark if isinstance(mark, tuple) else mark
