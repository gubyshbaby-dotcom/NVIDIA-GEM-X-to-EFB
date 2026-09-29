"""Building and finding the rig. The only place that creates bpy data."""

from __future__ import annotations

import bpy
from mathutils import Matrix

from efb.armature import Armature
from efb.matrix import Mat4
from efb.rig import COORD_BONE, Rig
from efb.rigdef import TOOL_JOINTS, limbs_from_json, tool_bind_prop
from efb.rigedit import SEAM_CONSTRAINT, seams

__all__ = ["to_matrix", "from_matrix", "build_rig", "find_rig", "rig_for",
           "fk_controls", "ik_controls", "seam_followers", "unbound_tools",
           "bind_keys", "bind_switches", "is_constrained", "on_ik", "inert_ik_bones",
           "seam_controls", "seam_shift", "action_curves", "curve_bags", "location_keys",
           "offset_location_keys", "local_delta", "imported_action",
           "ARMATURE_PROP", "FK_CONSTRAINT", "CLIP_PROP", "IK_SIDE"]

ARMATURE_PROP = "efb_armature"

FK_CONSTRAINT = "FK"


def to_matrix(m: Mat4) -> Matrix:
    return Matrix(m.rows())


def from_matrix(m) -> Mat4:
    return Mat4([v for row in m for v in row])


def build_rig(rig: Rig, name=None, collection=None):
    """A real Blender armature carrying the json rest matrices.

    Bone length is cosmetic - it never enters matrix_local - so it is taken from the
    child offsets purely so the rig is possible to look at. Every Epic Fight joint is a
    deform bone; Coord is not, which is also how the exporter tells them apart when the
    artist has added helpers of their own.
    """
    name = name or rig.armature.name
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    (collection or bpy.context.collection).objects.link(obj)
    obj[ARMATURE_PROP] = rig.armature.name

    view = bpy.context.view_layer
    prev = view.objects.active
    view.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in rig.bones():
        eb = data.edit_bones.new(bone)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, rig.bone_length(bone), 0.0)
        eb.matrix = to_matrix(rig.rest_model(bone))
        parent = rig.parent(bone)
        if parent:
            eb.parent = data.edit_bones[parent]
            eb.use_connect = False
        eb.use_deform = bone != COORD_BONE
    bpy.ops.object.mode_set(mode="OBJECT")
    if prev is not None:
        view.objects.active = prev

    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return obj


def find_rig(obj=None):
    """The armature object to work on: the argument, the active object, or the only
    armature in the scene."""
    if obj is not None and obj.type == "ARMATURE":
        return obj
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "ARMATURE":
        return active
    found = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(found) == 1:
        return found[0]
    raise RuntimeError("no armature selected and %d in the scene" % len(found))


def rig_for(obj, armature: Armature) -> Rig:
    """Pair a scene armature with its json. Coord counts only if the object has it."""
    return Rig(armature, coord=COORD_BONE in obj.pose.bones)


def fk_controls(obj) -> dict:
    """deform bone -> the FK control that drives it. Empty on a plain deform rig.

    On the control rig a deform bone is constraint-driven, so keys have to land on its
    control instead; the control's rest matrix is the joint's, which is what makes the
    delta come out the same.
    """
    deform = {b.name for b in obj.data.bones if b.use_deform}
    out = {}
    for pb in obj.pose.bones:
        if pb.name not in deform:
            continue
        con = pb.constraints.get(FK_CONSTRAINT)
        sub = getattr(con, "subtarget", "") if con is not None else ""
        if sub and sub in obj.pose.bones:
            out[pb.name] = sub
    return out


IK_SIDE = ("ik_upper", "ik_ctrl", "pole_ctrl", "ik_twist", "ik_lower")


def _limbs(obj):
    return limbs_from_json(obj.data.get("efb_limbs", "[]") if obj.data else "[]")


def _action(obj, action):
    if action is not None:
        return action
    return obj.animation_data.action if obj.animation_data else None


def on_ik(obj, limb, action=None) -> bool:
    """Whether this limb is solved anywhere in the action. The curve wins over the
    property: the property is only where the blend sits right now."""
    action = _action(obj, action)
    path = '["%s"]' % limb.prop
    for fc in (action_curves(action, obj) if action is not None else ()):
        if fc.data_path == path and len(fc.keyframe_points):
            return any(kp.co[1] > 0.0 for kp in fc.keyframe_points)
    return float(obj.get(limb.prop, 0.0)) > 0.0


def ik_side_bones(obj, limb) -> list:
    return [getattr(limb, f) for f in IK_SIDE
            if getattr(limb, f, "") and getattr(limb, f) in obj.pose.bones]


def inert_ik_bones(obj, action=None) -> set:
    """The IK-side bones of every limb this action never solves.

    An import bakes the handles of a limb it then leaves on FK, so those keys drive
    nothing. Folding them into the export would put timestamps in the file the clip never
    had, and widen the ladder the drift check walks.
    """
    return {n for limb in _limbs(obj) if not on_ik(obj, limb, action)
            for n in ik_side_bones(obj, limb)}


def ik_controls(obj, action=None) -> dict:
    """IK control -> the deform joints its chain moves, for the limbs this action solves.
    Empty on a plain deform rig.

    Without this a limb posed on IK exports as nothing at all: the track list is built
    from the joints that carry keys, and keys on an IK target reach their joints through
    a solver rather than through a constraint that names them.
    """
    out = {}
    for limb in _limbs(obj):
        if not on_ik(obj, limb, action):
            continue
        for ctrl in (limb.ik_ctrl, limb.pole_ctrl):
            if ctrl in obj.pose.bones:
                out.setdefault(ctrl, []).extend([limb.upper, limb.lower])
    return out


def detach_controls(obj) -> dict:
    """detach control -> the deform joints it carries. A key on the detach moves the whole
    limb in both modes - FK by parentage, IK through the twin - so the export folds its
    frames onto the limb's joints the way it does for an IK target."""
    out = {}
    for limb in _limbs(obj):
        detach = getattr(limb, "detach", "")
        if detach and detach in obj.pose.bones:
            out[detach] = [limb.upper, limb.lower]
    return out


def seam_followers(obj) -> dict:
    """folding deform bone -> the joint markers that slide off it, or {} when the seam is
    off. Without this the seam is viewport-only and every clip we export tears in game:
    the track list is built from the bones that carry keys, and nobody keys Elbow_* by
    hand.

    Read off the seam table and the live constraint. It used to be found by shape - a
    deform bone driven by another deform bone - which stopped working the moment the fold
    was read off a hidden copy of the lower bone instead of the lower bone itself.
    """
    out = {}
    for s in seams(obj.pose.bones.keys()):
        con = obj.pose.bones[s.marker].constraints.get(SEAM_CONSTRAINT)
        if con is not None and con.influence > 0.0 and s.marker not in out.get(s.lower, ()):
            out.setdefault(s.lower, []).append(s.marker)
    return out


def bind_keys(obj, action=None) -> dict:
    """tool joint -> the frames its bind state is keyed at, for the sockets that carry an
    animated bind. Read off the fcurve and not off the Keyed flag: the flag is what the
    toggle writes with, the curve is what the rig actually plays, so the export follows
    the curve whatever the flag says."""
    ad = obj.animation_data
    action = action or (ad.action if ad else None)
    if action is None:
        return {}
    curves = {fc.data_path: fc for fc in action_curves(action, obj)}
    out = {}
    for joint in TOOL_JOINTS:
        fc = curves.get('["%s"]' % tool_bind_prop(joint))
        if fc is None or not len(fc.keyframe_points):
            continue
        out[joint] = sorted({float(kp.co[0]) for kp in fc.keyframe_points})
    return out


def bind_switches(obj, action=None) -> dict:
    """tool joint -> the frames the export has to carry a key on because the bind steps
    there. A step is two samples, not one: Epic Fight lerps between the keys it is handed,
    so the frame before the switch has to be in the file or the cut becomes a ramp."""
    out = {}
    for joint, frames in bind_keys(obj, action).items():
        got = set()
        for f in frames:
            got.update((f - 1.0, f))
        out[joint] = sorted(got)
    return out


def unbound_tools(obj, action=None) -> dict:
    """tool joint -> every deform joint above it, for each tool whose bind is off.

    An unbound tool's delta is a counter-animation of its parent, so it is curved even
    when the parent's motion is not and its own keys do not describe it: measured, a
    forearm turning 80 degrees over 20 frames drifts the tool 0.0639 m when the tool is
    keyed only at the ends. Handing it every frame its parents move on is the fix the
    seam markers already use.

    A socket whose bind is KEYED counts as unbound whatever the property reads right now:
    it is free for part of the clip, and the free part needs the parent's frames.
    """
    keyed = bind_keys(obj, action)
    out = {}
    for joint in TOOL_JOINTS:
        if joint not in obj.pose.bones:
            continue
        if joint not in keyed and obj.get(tool_bind_prop(joint), 1):
            continue
        chain, bone = [], obj.data.bones[joint].parent
        while bone is not None:
            if bone.use_deform:
                chain.append(bone.name)
            bone = bone.parent
        out[joint] = chain
    return out


def is_constrained(obj, names) -> bool:
    """Whether matrix_basis still describes these bones. It does not once anything
    constrains them - a constrained deform bone reads identity there."""
    return any(obj.pose.bones[n].constraints for n in names if n in obj.pose.bones)




CLIP_PROP = "efb_clip"


def imported_action(obj):
    """The rig's action if it came out of an Epic Fight clip, else None.

    Only those already carry the game's slide in their marker tracks, so only those want
    the constraint's copy compensated. A hand-animated action does not, and taking the
    slide out of it would make the Seam Slide toggle change nothing anyone can see.
    """
    ad = obj.animation_data
    action = ad.action if ad else None
    return action if action is not None and action.get(CLIP_PROP) else None


def seam_controls(obj) -> dict:
    """marker deform bone -> the control its keys live on, for every seam this skeleton
    has. Unlike seam_followers this does not care whether the constraint is on, which is
    what a toggle needs: it has to know the pairs before and after it flips them."""
    ctrls = fk_controls(obj)
    out = {}
    for s in seams(obj.pose.bones.keys()):
        ctrl = ctrls.get(s.marker, s.marker)
        if ctrl in obj.pose.bones:
            out[s.marker] = ctrl
    return out


def local_delta(pb) -> Matrix:
    """The bone's evaluated transform against its own rest, in its parent's space."""
    pose, rest = pb.matrix, pb.bone.matrix_local
    if pb.parent is not None:
        pose = pb.parent.matrix.inverted() @ pose
        rest = pb.parent.bone.matrix_local.inverted() @ rest
    return rest.inverted() @ pose


def seam_shift(obj, pairs, frames=None):
    """{marker: {frame: Vector}} - how far the seam constraints slide each marker past
    what its control asks for. Measured, not predicted from the map, so it is exact
    whatever the constraint is set to and zero when it is off.

    `frames` None measures the current frame and files it under None.
    """
    out = {m: {} for m in pairs}
    if not pairs:
        return out

    def measure():
        for marker, ctrl in pairs.items():
            yield marker, (local_delta(obj.pose.bones[marker]).to_translation()
                           - obj.pose.bones[ctrl].matrix_basis.to_translation())

    if frames is None:
        for marker, v in measure():
            out[marker][None] = v
        return out
    scene = bpy.context.scene
    saved = scene.frame_current, scene.frame_subframe
    for frame in sorted(frames):
        whole = int(frame // 1)
        scene.frame_set(whole, subframe=float(frame - whole))
        for marker, v in measure():
            out[marker][round(frame, 3)] = v
    scene.frame_set(saved[0], subframe=saved[1])
    return out


def curve_bags(action, obj=None):
    """What holds an action's fcurves, whichever way this Blender lays them out. 4.4+ keeps
    them in a slot/layer/strip; older builds keep action.fcurves. Each has .fcurves, which
    is what a caller removing one needs."""
    if not (hasattr(action, "layers") and action.layers):
        return [action]
    slot = None
    if obj is not None and obj.animation_data is not None:
        slot = getattr(obj.animation_data, "action_slot", None)
    slot = slot or (action.slots[0] if action.slots else None)
    if slot is None:
        return []
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot)
            if bag:
                out.append(bag)
    return out


def action_curves(action, obj=None):
    return [fc for bag in curve_bags(action, obj) for fc in bag.fcurves]


def location_keys(action, obj, bone) -> set:
    """The frames one control's location track is keyed at."""
    path = 'pose.bones["%s"].location' % bpy.utils.escape_identifier(bone)
    return {round(float(kp.co[0]), 3)
            for fc in action_curves(action, obj) if fc.data_path == path
            for kp in fc.keyframe_points}


def offset_location_keys(obj, action, pairs, delta):
    """Add a per-key vector to each control's location track.

    Keys are matched on a rounded frame: a keyframe's time comes back as float32, so the
    double that wrote it does not compare equal to it.
    """
    if action is None:
        return
    for marker, ctrl in pairs.items():
        per_frame = delta.get(marker) or {}
        if not per_frame:
            continue
        path = 'pose.bones["%s"].location' % bpy.utils.escape_identifier(ctrl)
        for fc in action_curves(action, obj):
            if fc.data_path != path:
                continue
            for kp in fc.keyframe_points:
                v = per_frame.get(round(float(kp.co[0]), 3))
                if v is not None:
                    kp.co[1] += v[fc.array_index]
            fc.update()
    obj.update_tag()
