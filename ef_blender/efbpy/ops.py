"""Operators. Each one is a thin wrapper over a plain function."""

import math

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from efb import assets
from efb.armature import Armature
from efb.rigdef import (TOOL_BIND_CONSTRAINT, build_biped_rig, limbs_from_json,
                        tool_bind_prop, tool_holder, tool_keyed_prop, validate)

from . import body, smooth, snap
from .animscene import ARMATURE_PROP, action_curves
from .build import build_rig

CTRL_TOOL = "CTRL-"


def rig_limbs(rig):
    return limbs_from_json(rig.data.get("efb_limbs", "[]"))


def rig_limb(rig, key):
    for limb in rig_limbs(rig):
        if limb.key == key:
            return limb
    raise KeyError(key)


_LIMB_BONES = ("upper", "lower", "marker", "fk_upper", "fk_lower", "ik_upper",
               "ik_lower", "ik_ctrl", "pole_ctrl", "detach")


def limb_of(rig, bone_name):
    """Which limb a bone belongs to, so the sidebar can show the blend for whatever the
    animator just clicked. None for the spine, the root stack and the sockets."""
    for limb in rig_limbs(rig):
        if any(getattr(limb, a) == bone_name for a in _LIMB_BONES):
            return limb
    return None


BUILDS = (("biped", "Wide", "Steve proportions, 4px arms"),
          ("biped_slim_arm", "Slim", "Alex proportions, 3px arms"))


class EFB_OT_generate_rig(bpy.types.Operator):
    """Build the Epic Fight biped rig, its body mesh and the default skin"""
    bl_idname = "efb.generate_rig"
    bl_label = "Generate Epic Fight Rig"
    bl_options = {"REGISTER", "UNDO"}

    jar: StringProperty(name="Epic Fight jar", default="", subtype="FILE_PATH",
                        description="Optional. Blank uses the armature and body mesh "
                                    "bundled with the addon")
    variant: EnumProperty(name="Build", items=BUILDS, default="biped")
    entity: StringProperty(name="Entity", default="",
                           description="Armature name; blank uses the chosen build")
    body_mesh: BoolProperty(name="Body Mesh", default=True,
                            description="Build the skinned body and bind it to the rig")
    skin: StringProperty(name="Skin", default="", subtype="FILE_PATH",
                         description="Optional 64x64 skin png; blank uses the default")

    def execute(self, context):
        entity = self.entity or self.variant
        try:
            arm = assets.armature(entity, self.jar or None)
            mesh = assets.body_mesh(entity, self.jar or None) if self.body_mesh else None
        except Exception as exc:
            self.report({"ERROR"}, "cannot read %s: %s" % (entity, exc))
            return {"CANCELLED"}
        return build_and_report(self, context, arm, mesh, entity,
                                self.skin or assets.skin_path(entity))


RIG_NAMES = {"biped": "EF-biped", "biped_slim_arm": "EF-biped-slim"}


def build_default_rig(context, arm: Armature, mesh=None, variant=None, skin=None):
    """Generate's rig as a plain function - controls, bodies and all. Returns (obj, note).

    Split out of the operator so an import into an empty scene can spawn exactly the rig
    the Generate button makes, instead of a second, thinner one.
    """
    key = variant or arm.name
    rigdef = build_biped_rig(arm, name=RIG_NAMES.get(key, "EF-" + str(key)))
    errors = validate(rigdef)
    if errors:
        raise RuntimeError("rig definition is invalid: %s" % errors[0])
    obj = build_rig(rigdef, context=context)
    obj[ARMATURE_PROP] = variant or arm.name
    from . import rigedit
    rigedit.apply_all(obj)
    note = "%d bones, %d deform" % (len(rigdef.bones), len(rigdef.deform_bones()))
    if mesh is not None:
        me = body.build_body(mesh, obj, skin_path=skin, variant=variant, context=context)
        smoothed = smooth.build_smooth_body(me, obj, context=context)
        note += ", body %d verts%s, authoring body %d" % (
            len(me.data.vertices), " textured" if me.data.materials else "",
            len(smoothed.data.vertices))
        context.view_layer.objects.active = obj
    return obj, note


def build_and_report(op, context, arm: Armature, mesh=None, variant=None, skin=None):
    try:
        obj, note = build_default_rig(context, arm, mesh, variant, skin)
    except RuntimeError as exc:
        op.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    op.report({"INFO"}, "built %s: %s" % (obj.name, note))
    return {"FINISHED"}


def named_rig(op, context):
    """The rig an operator was pointed at, or the active one. None if it is not a rig."""
    obj = bpy.data.objects.get(op.rig) if op.rig else context.object
    return obj if snap.is_rig(obj) else None


class EFB_OT_set_skin(bpy.types.Operator, ImportHelper):
    """Point this rig at a different skin png. Both its bodies move together and its
    pose, action and everything else are kept - the other rigs in the scene are not
    touched"""
    bl_idname = "efb.set_skin"
    bl_label = "Change Skin"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})

    rig: StringProperty(name="Rig", default="",
                        description="Which rig to reskin. Blank uses the active one")
    default_skin: BoolProperty(name="Bundled Default", default=False,
                               description="Ignore the file and go back to the skin this "
                                           "build ships with")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def invoke(self, context, event):
        if self.default_skin:
            return self.execute(context)
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        rig = named_rig(self, context)
        if rig is None:
            self.report({"ERROR"}, "%r is not an Epic Fight rig" % self.rig)
            return {"CANCELLED"}
        try:
            material, bodies = body.set_skin(
                rig, "" if self.default_skin else self.filepath, context)
        except (OSError, RuntimeError) as exc:
            self.report({"ERROR"}, "cannot skin %s: %s" % (rig.name, exc))
            return {"CANCELLED"}
        image = body.skin_image_of(material)
        self.report({"INFO"}, "%s: %d bodies now wear %s"
                    % (rig.name, len(bodies), image.name if image else "no skin"))
        return {"FINISHED"}


class _SnapBase(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}
    limb: StringProperty(name="Limb", default="arm_r")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object) and context.mode == "POSE"


class EFB_OT_snap_ik_to_fk(_SnapBase):
    """Put the IK control and pole on the current FK pose and switch to IK"""
    bl_idname = "efb.snap_ik_to_fk"
    bl_label = "Snap IK to FK"

    def execute(self, context):
        rig = context.object
        snap.snap_ik_to_fk(rig, rig_limb(rig, self.limb), context)
        return {"FINISHED"}


class EFB_OT_snap_fk_to_ik(_SnapBase):
    """Put the FK chain on the current IK solution and switch to FK"""
    bl_idname = "efb.snap_fk_to_ik"
    bl_label = "Snap FK to IK"

    def execute(self, context):
        rig = context.object
        snap.snap_fk_to_ik(rig, rig_limb(rig, self.limb), context)
        return {"FINISHED"}


class EFB_OT_reset_pose(bpy.types.Operator):
    """Clear every control back to the Epic Fight rest pose"""
    bl_idname = "efb.reset_pose"
    bl_label = "Reset Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def execute(self, context):
        snap.reset_pose(context.object, context)
        return {"FINISHED"}


class EFB_OT_set_limb_mode(bpy.types.Operator):
    """Switch a limb to IK or FK without moving anything"""
    bl_idname = "efb.set_limb_mode"
    bl_label = "Set Limb Mode"
    bl_options = {"REGISTER", "UNDO"}

    limb: StringProperty(name="Limb", default="arm_r")
    mode: EnumProperty(name="Mode", items=(("FK", "FK", ""), ("IK", "IK", "")),
                       default="FK")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        snap.set_mode(rig, rig_limb(rig, self.limb), self.mode == "IK", context)
        return {"FINISHED"}


BIND_INTERPOLATION = "CONSTANT"


def bind_curve(rig, joint):
    """The fcurve animating one socket's bind, or None."""
    ad = rig.animation_data
    action = ad.action if ad else None
    if action is None:
        return None
    path = '["%s"]' % tool_bind_prop(joint)
    return next((fc for fc in action_curves(action, rig) if fc.data_path == path), None)


def _key_bind(rig, joint, frame, value):
    """One key on the bind channel, at the value we mean rather than at whatever the
    property happens to read: once the channel is animated the id property is overwritten
    from the curve on every evaluation, so an insert that reads it back records the old
    state instead of the new one.
    """
    if bind_curve(rig, joint) is None:
        rig.keyframe_insert('["%s"]' % tool_bind_prop(joint), frame=frame)
    fc = bind_curve(rig, joint)
    if fc is None:
        return
    fc.keyframe_points.insert(float(frame), float(value))
    for kp in fc.keyframe_points:
        kp.interpolation = BIND_INTERPOLATION
    fc.update()


def _key_socket(pb, frame):
    for channel in ("location", "rotation_quaternion", "scale"):
        pb.keyframe_insert(channel, frame=frame)


def set_tool_bind(rig, joint, bound, context=None, keep_place=True, keyed=None,
                  frame=None):
    """Bind or unbind one Tool socket, holding its world pose across the change.

    Without the hold the control snaps back to the hand's own rest the moment the Child Of
    lets go, because the holder it hangs off is unparented. The write goes through
    set_matrix for the same reason a space switch does - the control may carry a rest
    override, and matrix_basis alone would not take it out.

    Keyed, the state and the hold both become keys, on both sides of the switch, so the
    correction lands in the action instead of being thrown away at the next frame change.
    That is the whole ordering rule: keys win, and the hold is one of them.
    """
    from .restpose import set_matrix
    context = context or bpy.context
    pb = rig.pose.bones[CTRL_TOOL + joint]
    if keyed is None:
        keyed = bool(rig.get(tool_keyed_prop(joint), 0))
    frame = context.scene.frame_current if frame is None else frame
    was = 1 if rig.get(tool_bind_prop(joint), 1) else 0
    before = pb.matrix.copy() if keep_place else None
    if keyed:
        _key_bind(rig, joint, frame - 1, was)
        _key_socket(pb, frame - 1)
        _key_bind(rig, joint, frame, 1 if bound else 0)
    rig[tool_bind_prop(joint)] = 1 if bound else 0
    rig.update_tag()
    context.view_layer.update()
    if before is not None:
        set_matrix(pb, before)
        if keyed:
            _key_socket(pb, frame)
        context.view_layer.update()
    elif keyed:
        _key_socket(pb, frame)


RETURN_TOLERANCE = (1e-6, 1e-3)


def place_gap(a, b):
    """(metres, degrees) between two world matrices. Position and orientation are the two
    units a socket is judged in; the raw matrix gap is neither."""
    off = (a.translation - b.translation).length
    turn = math.degrees(a.to_quaternion().rotation_difference(b.to_quaternion()).angle)
    return off, turn


def bind_hand(rig, joint):
    """The joint this socket's holder is bound to, read off the Child Of rather than off a
    table, so a rig that names its hands differently still returns to the right one."""
    holder = rig.pose.bones.get(tool_holder(joint))
    con = holder.constraints.get(TOOL_BIND_CONSTRAINT) if holder else None
    return con.subtarget if con is not None and con.subtarget else ""


def hand_placement(rig, joint):
    """Where the hand puts this socket: Epic Fight's own Tool_* offset, carried by the live
    pose of the hand it hangs under. This is what a bound, unposed socket evaluates to, so
    landing on it is landing on the frame the game holds the weapon in - rotation included.
    """
    from . import tool
    hand = bind_hand(rig, joint)
    if not hand or hand not in rig.pose.bones:
        raise KeyError("%s has no bound hand for %s" % (rig.name, joint))
    return tool.deform_matrix(rig, hand) @ rig.data.bones[CTRL_TOOL + joint].matrix_local


def return_tool(rig, joint, context=None, keyed=None, frame=None):
    """Put one Tool socket back on the placement its hand dictates, and bind it again.

    The other half of the bind toggle. Unbinding holds the socket's world pose so the
    weapon stays where it is while the arm moves; this walks it back - position AND
    rotation - to the place the hand would be holding it if nobody had touched it.

    Returns (metres, degrees) travelled, or None when it was already there and bound, in
    which case nothing is written: no move, no key, not even a bind key.
    """
    from .restpose import set_matrix
    context = context or bpy.context
    pb = rig.pose.bones[CTRL_TOOL + joint]
    want = hand_placement(rig, joint)
    bound = bool(rig.get(tool_bind_prop(joint), 1))
    off, turn = place_gap(pb.matrix, want)
    if bound and off <= RETURN_TOLERANCE[0] and turn <= RETURN_TOLERANCE[1]:
        return None
    if keyed is None:
        keyed = bool(rig.get(tool_keyed_prop(joint), 0))
    frame = context.scene.frame_current if frame is None else frame
    if not bound:
        if keyed:
            _key_bind(rig, joint, frame - 1, 0)
            _key_socket(pb, frame - 1)
            _key_bind(rig, joint, frame, 1)
        rig[tool_bind_prop(joint)] = 1
        rig.update_tag()
        context.view_layer.update()
    set_matrix(pb, want)
    if keyed:
        _key_socket(pb, frame)
    context.view_layer.update()
    return off, turn


class EFB_OT_return_tool(bpy.types.Operator):
    """Put this Tool socket back where the hand holds it, position and rotation both, and
    bind it to the hand again. A socket that is already bound and untouched is left alone"""
    bl_idname = "efb.return_tool"
    bl_label = "Return Tool to Hand"
    bl_options = {"REGISTER", "UNDO"}

    joint: StringProperty(name="Joint", default="Tool_R")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def execute(self, context):
        from . import tool
        rig = context.object
        if CTRL_TOOL + self.joint not in rig.pose.bones:
            self.report({"ERROR"}, "%s has no %s socket" % (rig.name, self.joint))
            return {"CANCELLED"}
        try:
            moved = return_tool(rig, self.joint, context)
        except KeyError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if moved is None:
            self.report({"INFO"}, "%s is already where the hand holds it" % self.joint)
            return {"FINISHED"}
        pb = rig.pose.bones[CTRL_TOOL + self.joint]
        self.report({"INFO"}, "%s back on the hand: %.3f m, %.1f deg, blade axis %s"
                    % (self.joint, moved[0], moved[1],
                       ", ".join("%+.2f" % c for c in tool.reach_axis(pb.matrix))))
        if not rig.get(tool_keyed_prop(self.joint), 0) and \
                bind_curve(rig, self.joint) is not None:
            self.report({"WARNING"},
                        "%s's bind is animated and Keyed is off, so this return is not in "
                        "the action - the next frame change plays the old keys back over "
                        "it" % self.joint)
        return {"FINISHED"}


class EFB_OT_set_tool_bind(bpy.types.Operator):
    """Bind or unbind this Tool socket from the hand. Unbound it holds still while the arm
    moves, and the exported Tool_* track carries the counter-animation"""
    bl_idname = "efb.set_tool_bind"
    bl_label = "Set Tool Bind"
    bl_options = {"REGISTER", "UNDO"}

    joint: StringProperty(name="Joint", default="Tool_R")
    bound: BoolProperty(name="Bound", default=True)

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        if CTRL_TOOL + self.joint not in rig.pose.bones:
            self.report({"ERROR"}, "%s has no %s socket" % (rig.name, self.joint))
            return {"CANCELLED"}
        set_tool_bind(rig, self.joint, self.bound, context)
        self.report({"INFO"}, "%s %s the hand"
                    % (self.joint, "follows" if self.bound else "is free of"))
        return {"FINISHED"}


CLASSES = (EFB_OT_generate_rig, EFB_OT_set_skin, EFB_OT_snap_ik_to_fk,
           EFB_OT_snap_fk_to_ik, EFB_OT_reset_pose, EFB_OT_set_limb_mode,
           EFB_OT_set_tool_bind, EFB_OT_return_tool)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
