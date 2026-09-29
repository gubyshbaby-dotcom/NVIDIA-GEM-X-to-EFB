"""Put an older rig's IK handles and poles under the root, in place.

Rigs built before version 11 left the four IK handles and the four pole holders loose in
armature space, which is what the reference rig does. Moving CTRL-Master turned the whole
figure and left eight bones standing: the four handles, their four roll helpers, and the
poles with them. New rigs are built with the master carrying all of it. This is the way to
put an existing one on the same footing without regenerating it and rebaking through the
Epic Fight document.

What it does to keys is the whole question, and the answer has three cases:

  the master is at rest and unkeyed   Nothing to compensate. A key on a loose handle is an
                                      armature-space place; the same key under a master
                                      that is an identity is the same place. Every key
                                      keeps its exact meaning and none of them is touched.
  the master is posed but unkeyed     The handles would inherit that pose. The controls are
                                      put back on the world matrices they were producing,
                                      one write each. Exact.
  the master is keyed                 The handles would inherit a moving parent, and no
                                      rewrite of their existing keys reproduces that
                                      between keys. The eight affected controls are baked
                                      over the action's range, one linear key per frame.
                                      The world path survives; their curve shape does not.

Nothing else on the rig is touched: the deform rests, the other controls, their keys and
the export are all left where they were.
"""

import bpy
from bpy.props import BoolProperty
from mathutils import Matrix

from efb.rigdef import POLE_ROOT_SPACE

from . import snap
from .ops import rig_limbs
from .restpose import keyed_controls, set_matrix

__all__ = ["MASTER", "loose_handles", "loose_poles", "needs_carrying", "adopt"]

MASTER = "CTRL-Master"

AT_REST = 1e-6


def _parent_of(rig, name):
    bone = rig.data.bones.get(name)
    return bone.parent.name if bone is not None and bone.parent else None


def _master_carries(rig, name) -> bool:
    """Whether the bone parent chain reaches the master. A version 12 handle hangs off its
    limb's detach twin, which hangs off the master; older carried ones hang off it
    directly; a loose one reaches nothing."""
    seen = set()
    cur = _parent_of(rig, name)
    while cur is not None and cur not in seen:
        if cur == MASTER:
            return True
        seen.add(cur)
        cur = _parent_of(rig, cur)
    return False


def loose_handles(rig):
    """IK handles the master is not carrying."""
    if MASTER not in rig.pose.bones:
        return []
    return [l.ik_ctrl for l in rig_limbs(rig)
            if l.ik_ctrl in rig.pose.bones and not _master_carries(rig, l.ik_ctrl)]


def loose_poles(rig):
    """Limbs whose pole holder has no root space, so with pole-follow off the pole stands
    still in armature space and root motion drives away from it."""
    if MASTER not in rig.pose.bones:
        return []
    return [l for l in rig_limbs(rig)
            if l.pole_space in rig.pose.bones
            and POLE_ROOT_SPACE not in rig.pose.bones[l.pole_space].constraints]


def needs_carrying(rig) -> bool:
    return bool(loose_handles(rig)) or bool(loose_poles(rig))


def master_moves(rig) -> bool:
    """Is the root doing anything the handles would start following - a pose off its rest,
    or a key anywhere in the action."""
    pb = rig.pose.bones.get(MASTER)
    if pb is None:
        return False
    ident = Matrix.Identity(4)
    m = pb.matrix_basis
    if any(abs(m[r][c] - ident[r][c]) > AT_REST for r in range(4) for c in range(4)):
        return True
    return bool(keyed_controls(rig, [MASTER]))


def _action(rig):
    ad = rig.animation_data
    return ad.action if ad else None


def _frames(rig, context):
    action = _action(rig)
    if action is not None and getattr(action, "frame_range", None):
        lo, hi = action.frame_range
        if hi > lo:
            return list(range(int(round(lo)), int(round(hi)) + 1))
    scene = context.scene
    return list(range(scene.frame_start, scene.frame_end + 1))


def _sample(rig, names, frames, context):
    """{frame or None: {bone: world matrix}} before anything is reparented."""
    if not frames:
        context.view_layer.update()
        return {None: {n: rig.pose.bones[n].matrix.copy() for n in names}}
    was = context.scene.frame_current
    held = {}
    for frame in frames:
        context.scene.frame_set(frame)
        held[frame] = {n: rig.pose.bones[n].matrix.copy() for n in names}
    context.scene.frame_set(was)
    return held


def _write(rig, names, want, context, key=False, frame=None):
    """Put every named control back on its held matrix, worst residual returned.

    Twice over: the poles hang off holders whose own space switch reads the chord, which
    reads a handle written in the same pass, so one sweep leaves the second half chasing
    the first.
    """
    worst = 0.0
    for _ in range(2):
        for name in names:
            set_matrix(rig.pose.bones[name], want[name])
        rig.update_tag()
        context.view_layer.update()
    for name in names:
        pb = rig.pose.bones[name]
        worst = max(worst, (pb.matrix.translation - want[name].translation).length)
        if key:
            turn = ("rotation_quaternion" if pb.rotation_mode == "QUATERNION" else
                    "rotation_axis_angle" if pb.rotation_mode == "AXIS_ANGLE" else
                    "rotation_euler")
            for channel in ("location", turn, "scale"):
                pb.keyframe_insert(channel, frame=frame)
    return worst


def _reparent(rig, names, context):
    was = rig.mode
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for name in names:
            rig.data.edit_bones[name].parent = rig.data.edit_bones[MASTER]
            rig.data.edit_bones[name].use_connect = False
    finally:
        bpy.ops.object.mode_set(mode=was)
    context.view_layer.update()


def _add_root_space(rig, limb):
    """The pole holder's second Child Of, driven complementary to the chord's.

    Seated against the master's REST and not against its pose: that is what a fresh rig
    does, and seating it against a posed master would bake the pose into the constraint.
    """
    from .rigedit import _influence_path, _prop_driver
    holder = rig.pose.bones[limb.pole_space]
    con = holder.constraints.new("CHILD_OF")
    con.name = POLE_ROOT_SPACE
    con.target = rig
    con.subtarget = MASTER
    con.influence = 1.0 - float(rig.get(limb.pole_follow, 1.0))
    con.inverse_matrix = rig.data.bones[MASTER].matrix_local.inverted()
    _prop_driver(rig, _influence_path(limb.pole_space, POLE_ROOT_SPACE), "1 - v",
                 limb.pole_follow)


def adopt(rig, context=None) -> dict:
    """Hand the loose handles and poles to the master, keeping what is on screen and in the
    action. Returns what was done, for the operator to say out loud."""
    context = context or bpy.context
    handles, poles = loose_handles(rig), loose_poles(rig)
    out = {"handles": handles, "poles": [l.label for l in poles], "baked": [],
           "frames": 0, "worst": 0.0}
    if not handles and not poles:
        return out

    watched = handles + [l.pole_ctrl for l in poles if l.pole_ctrl in rig.pose.bones]
    moving = master_moves(rig)
    frames = _frames(rig, context) if moving and keyed_controls(rig, [MASTER]) else []
    held = _sample(rig, watched, frames, context) if moving else {}

    _reparent(rig, handles, context)
    for limb in poles:
        _add_root_space(rig, limb)
    rig.update_tag()
    context.view_layer.update()

    drifted = _drifted(rig, watched, held, frames, context) if held else []
    if drifted and not frames:
        out["worst"] = _write(rig, drifted, held[None], context)
    elif drifted:
        was = context.scene.frame_current
        for frame in frames:
            context.scene.frame_set(frame)
            out["worst"] = max(out["worst"],
                               _write(rig, drifted, held[frame], context, key=True,
                                      frame=frame))
        context.scene.frame_set(was)
        for fc in _bake_curves(rig, drifted):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
        out["baked"], out["frames"] = sorted(drifted), len(frames)
    rig.update_tag()
    context.view_layer.update()
    return out


def _drifted(rig, names, held, frames, context):
    """Which of these controls the change actually moved, over the frames it was held on."""
    if not frames:
        context.view_layer.update()
        return [n for n in names
                if (rig.pose.bones[n].matrix.translation
                    - held[None][n].translation).length > AT_REST]
    was, out = context.scene.frame_current, set()
    for frame in frames:
        context.scene.frame_set(frame)
        out |= {n for n in names
                if (rig.pose.bones[n].matrix.translation
                    - held[frame][n].translation).length > AT_REST}
    context.scene.frame_set(was)
    return [n for n in names if n in out]


def _bake_curves(rig, names):
    from .animscene import action_curves
    action = _action(rig)
    if action is None:
        return []
    paths = tuple('pose.bones["%s"].' % bpy.utils.escape_identifier(n) for n in names)
    return [fc for fc in action_curves(action, rig)
            if fc.data_path.startswith(paths)]


class EFB_OT_root_carry(bpy.types.Operator):
    """Hand this rig's IK handles and poles to the root control, so moving CTRL-Master
    moves the whole figure. Rigs built before version 11 left them loose in armature space
    and the hands, the feet and the poles stayed behind"""
    bl_idname = "efb.root_carry"
    bl_label = "Root Carries Everything"
    bl_options = {"REGISTER", "UNDO"}

    keep_animation: BoolProperty(
        name="Keep the animation", default=True,
        description="Hold the world path of every control this moves. Off, only the "
                    "parenting changes and a keyed root motion is applied twice")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        if not needs_carrying(rig):
            self.report({"INFO"}, "the root already carries everything on %s" % rig.name)
            return {"FINISHED"}
        moving = master_moves(rig)
        out = (adopt(rig, context) if self.keep_animation
               else _adopt_structure_only(rig, context))
        self.report({"INFO"}, "%d handle(s) and %d pole(s) now hang off %s"
                    % (len(out["handles"]), len(out["poles"]), MASTER))
        if out["baked"]:
            self.report({"WARNING"},
                        "%s carried keys and the root moves, so they were baked over %d "
                        "frames: the world path is kept, the curve shape is not"
                        % (", ".join(out["baked"]), out["frames"]))
        elif moving and self.keep_animation:
            self.report({"INFO"}, "the root is posed, so the moved controls were put back "
                                  "on the places they were producing")
        return {"FINISHED"}


def _adopt_structure_only(rig, context):
    """The parenting and nothing else, for a file whose root never moves."""
    handles, poles = loose_handles(rig), loose_poles(rig)
    _reparent(rig, handles, context)
    for limb in poles:
        _add_root_space(rig, limb)
    rig.update_tag()
    context.view_layer.update()
    return {"handles": handles, "poles": [l.label for l in poles], "baked": [],
            "frames": 0, "worst": 0.0}


CLASSES = (EFB_OT_root_carry,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
