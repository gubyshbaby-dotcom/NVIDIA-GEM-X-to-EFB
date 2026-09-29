"""IK/FK snapping. Plain functions first, operators around them, so the headless gate
can drive the same code path the buttons do."""

import math

import bpy
from mathutils import Matrix, Quaternion

from efb.rigdef import RIG_ID


def is_rig(obj):
    return (obj is not None and obj.type == "ARMATURE"
            and obj.data.get("rig_id") == RIG_ID)


def _update(context):
    (context or bpy.context).view_layer.update()


def _set_blend(rig, prop, value, context):
    """Writing an id property does not tag the depsgraph, so the drivers reading it would
    not re-run until something else did."""
    rig[prop] = value
    rig.update_tag()
    _update(context)


def _write(pose_bone, matrix):
    from .restpose import set_matrix
    set_matrix(pose_bone, matrix)


def _keyed_bones(rig, limb):
    names = [limb.fk_upper, limb.fk_lower, limb.ik_ctrl, limb.pole_ctrl, limb.ik_twist]
    return [rig.pose.bones[n] for n in names if n in rig.pose.bones]


def _key(rig, limb, frame):
    for pb in _keyed_bones(rig, limb):
        pb.keyframe_insert("location", frame=frame)
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        pb.keyframe_insert("scale", frame=frame)
    rig.keyframe_insert('["%s"]' % limb.prop, frame=frame)


def _rides(rig, carrier, rider):
    """Where `rider` would sit if it hung off `carrier` at its own rest offset.

    Two helper bones per limb used to stand here holding exactly this. They were rest-pose
    markers and nothing else, so the same matrix comes out of the rests the rig already
    carries: a bone's pose is its parent's pose times the rest offset between them.
    """
    bones = rig.data.bones
    return (rig.pose.bones[carrier].matrix
            @ bones[carrier].matrix_local.inverted() @ bones[rider].matrix_local)


def snap_ik_to_fk(rig, limb, context=None, keyframe=None, frame=None):
    """Move the IK control and pole onto the current FK pose, then switch to IK."""
    context = context or bpy.context
    frame = context.scene.frame_current if frame is None else frame
    keyframe = bool(rig.get("snap_keyframe")) if keyframe is None else keyframe
    pb = rig.pose.bones

    if keyframe:
        _key(rig, limb, frame - 1)
    _update(context)
    ik_target = _rides(rig, limb.fk_lower, limb.ik_ctrl)
    pole_target = _rides(rig, limb.marker, limb.pole_ctrl)
    fk_upper = pb[limb.upper].matrix.copy()

    _clear_twist(pb, limb)
    _write(pb[limb.ik_ctrl], ik_target)
    _update(context)
    pole_target.translation = _pole_position(pb, limb, ik_target.translation,
                                             pole_target.translation)
    _write(pb[limb.pole_ctrl], pole_target)
    _update(context)

    _set_blend(rig, limb.prop, 1.0, context)
    _carry_twist(pb, limb, fk_upper, context)
    if keyframe:
        _key(rig, limb, frame)


def _clear_twist(pb, limb):
    if limb.ik_twist in pb:
        pb[limb.ik_twist].rotation_quaternion = Quaternion()


def _carry_twist(pb, limb, fk_upper, context):
    """The one thing the solve cannot reproduce: FK's turn about the upper limb's own
    axis. Target and pole fix the direction and the plane and leave no room for it, so
    what is left between the solved upper limb and the FK one is a pure twist - and only
    its twist part is taken, so a target the chain could not reach shows as a limb off its
    elbow instead of being papered over.
    """
    if limb.ik_twist not in pb:
        return
    residual = pb[limb.upper].matrix.to_3x3().inverted() @ fk_upper.to_3x3()
    q = residual.to_quaternion()
    if q.w < 0.0:
        q.negate()
    pb[limb.ik_twist].rotation_quaternion = Quaternion(
        (0.0, 1.0, 0.0), 2.0 * math.atan2(q.y, q.w))
    _update(context)


def _pole_position(pb, limb, tip, fallback):
    """Where the pole has to sit for the IK to reproduce the FK bend: out from the middle
    joint, away from the line the chain spans, at the rest pole distance. A straight chain
    gives no direction, so it falls back to the marker joint's own heading."""
    root = pb[limb.fk_upper].matrix.translation
    middle = pb[limb.fk_lower].matrix.translation
    distance = (fallback - middle).length
    span = tip - root
    if span.length_squared < 1e-12 or distance < 1e-6:
        return fallback
    out = middle - (root + span * ((middle - root).dot(span) / span.length_squared))
    if out.length < 1e-4:
        return fallback
    return middle + out.normalized() * distance


def snap_fk_to_ik(rig, limb, context=None, keyframe=None, frame=None):
    """Move the FK chain onto the current IK solution, then switch to FK."""
    context = context or bpy.context
    frame = context.scene.frame_current if frame is None else frame
    keyframe = bool(rig.get("snap_keyframe")) if keyframe is None else keyframe
    pb = rig.pose.bones

    if keyframe:
        _key(rig, limb, frame - 1)
    _update(context)
    upper = pb[limb.upper].matrix.copy()
    lower = pb[limb.lower].matrix.copy()

    _write(pb[limb.fk_upper], upper)
    _update(context)
    _write(pb[limb.fk_lower], lower)
    _update(context)

    _set_blend(rig, limb.prop, 0.0, context)
    _clear_twist(pb, limb)
    if keyframe:
        _key(rig, limb, frame)


def set_mode(rig, limb, ik, context=None):
    """Flip a limb between FK and IK without moving anything."""
    _set_blend(rig, limb.prop, 1.0 if ik else 0.0, context)


def reset_pose(rig, context=None):
    """Clears every control back to rest. Deform bones are constrained, so they follow.

    Rest-override bones are skipped, so "rest" means whatever the animator last set it to;
    clearing them would silently throw the override away.
    """
    from .restpose import is_override
    for pb in rig.pose.bones:
        if not is_override(pb):
            pb.matrix_basis = Matrix.Identity(4)
    _update(context)
