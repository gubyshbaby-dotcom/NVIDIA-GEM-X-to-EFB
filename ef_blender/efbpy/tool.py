"""Hang an object off a Tool socket, so a weapon can be looked at in the pose it will
hold in game.

Two kinds of asset, and the difference is not cosmetic. A skinned weapon is authored in
the biped's own rest space and binds by vertex group, so an Armature modifier is the whole
attach. Everything else is a rigid item and rides one bone.

Both classes are already baked into the armature's rest space, so both reduce to the same
placement: the object origin sits on the grip and efwa_rest_matrix is the translation that
puts it back where the bake left it. efwa_bone_matrix is the item-local frame, not a
placement - re-applying it here would transform vertices that already carry it. These are
efwa.attach's rules; this module has to agree with them exactly, gated by weapon-attach.

    efwa_class            item_model or skinned_mesh
    efwa_joint            the Epic Fight joint a rigid item was baked against
    efwa_rest_matrix      origin -> bake space, flat 16 floats
    efwa_joint_matrix     that joint's rest matrix, as the bake saw it
    efwa_bone_matrix      the item-local frame (_offhand, _back too), for swapping hands

An object with none of that is placed with its own origin on the socket.
"""

import bpy
from bpy.props import StringProperty
from mathutils import Matrix

from efb.rigedit import TOOL_AXIS, TOOL_AXIS_SIGN, TOOL_SOCKETS

from . import snap

__all__ = ["attach", "detach", "attached_to", "socket_items", "deform_matrix",
           "reach_axis"]

CLASS_PROP = "efwa_class"
SKINNED_CLASS = "skinned_mesh"
JOINT_PROP = "efwa_joint"
JOINT_MATRIX = "efwa_joint_matrix"
BONE_MATRIX = "efwa_bone_matrix"
REST_MATRIX = "efwa_rest_matrix"

SOCKET_PROP = "efb_socket"

_SWAP = {"Tool_L": BONE_MATRIX + "_offhand", "Chest": BONE_MATRIX + "_back"}

IDENTITY = Matrix.Identity(4)


def socket_items(rig=None):
    names = [n for n in TOOL_SOCKETS if rig is None or n in rig.pose.bones]
    return [(n, n.replace("_R", " (right)").replace("_L", " (left)"), "") for n in names]


def _matrix(obj, key):
    """Library matrices are flat 16 floats; older assets stored four nested rows."""
    raw = obj.get(key) if key else None
    if raw is None:
        return None
    try:
        flat = [float(x) for x in raw]
    except TypeError:
        try:
            rows = [[float(x) for x in r] for r in raw]
        except TypeError:
            return None
        return Matrix(rows) if len(rows) == 4 and all(len(r) == 4 for r in rows) else None
    return Matrix([flat[0:4], flat[4:8], flat[8:12], flat[12:16]]) if len(flat) == 16 \
        else None


def _rest(obj):
    """Object origin -> the space the asset was baked in. Identity for anything else."""
    m = _matrix(obj, REST_MATRIX)
    return m if m is not None else IDENTITY


def _from_library(obj) -> bool:
    return any(k in obj.keys() for k in (CLASS_PROP, JOINT_PROP, REST_MATRIX))


def deform_matrix(rig, bone):
    """Blender's own deform matrix for a bone, identity at rest. matrix_local rather than
    efwa_joint_matrix, so a bone-parented item and an Armature-modified mesh move by the
    same matrix - Blender orthonormalises Tool_R/Tool_L and the two differ otherwise."""
    return rig.pose.bones[bone].matrix @ rig.data.bones[bone].matrix_local.inverted()


def reach_axis(matrix):
    """Which way a weapon held on this socket points, from the socket's world matrix. One
    axis of the socket bone's own frame - the same constant the widget and the stick read,
    because the three disagreeing is how the widget came to point backwards."""
    return (matrix.to_3x3().col[TOOL_AXIS] * TOOL_AXIS_SIGN).normalized()


def _skinned(obj, rig) -> bool:
    """A weapon mesh bound by joint name, the way Epic Fight's own weapon models are."""
    kind = obj.get(CLASS_PROP)
    if kind:
        return kind == SKINNED_CLASS
    if obj.type != "MESH" or not obj.vertex_groups:
        return False
    deform = {b.name for b in rig.data.bones if b.use_deform}
    return any(g.name in deform for g in obj.vertex_groups)


def unbound(obj, rig):
    """Library joints this rig cannot supply. efwa.attach refuses class 2 over these, and
    a bind that quietly drops half a weapon's groups is worse than an error."""
    if obj.get(CLASS_PROP) != SKINNED_CLASS:
        return []
    return [g.name for g in obj.vertex_groups if g.name not in rig.data.bones]


def _swap(obj, rig, bone):
    """Rebuild a main-hand item's frame for the offhand or the back, or None. Undo the
    bake's own frame, redo it at `bone` with that socket's recorded placement."""
    sibling = _matrix(obj, _SWAP.get(bone))
    joint = _matrix(obj, JOINT_MATRIX)
    placement = _matrix(obj, BONE_MATRIX)
    if sibling is None or joint is None or placement is None:
        return None
    return (rig.data.bones[bone].matrix_local @ sibling @ (joint @ placement).inverted())


def attached_to(obj):
    """(rig, socket) when this object is already on a socket, else (None, "")."""
    if obj is None or obj.parent is None or obj.parent_type != "BONE":
        mod = next((m for m in getattr(obj, "modifiers", ())
                    if m.type == "ARMATURE" and m.object), None) if obj else None
        return (mod.object, obj.get(SOCKET_PROP, "")) if mod else (None, "")
    return obj.parent, obj.parent_bone


def detach(obj):
    """Off the socket, left where it is on screen."""
    world = obj.matrix_world.copy()
    for mod in [m for m in obj.modifiers if m.type == "ARMATURE"
                and m.name == "EFB Tool"]:
        obj.modifiers.remove(mod)
    obj.parent = None
    obj.parent_bone = ""
    obj.parent_type = "OBJECT"
    obj.matrix_world = world


def attach(rig, obj, bone, context=None):
    """Put `obj` on `bone`. Returns the kind of attach that was used."""
    ctx = context or bpy.context
    if bone not in rig.pose.bones:
        raise ValueError("%s has no bone %r" % (rig.name, bone))
    if obj is rig:
        raise ValueError("a rig cannot be its own weapon")
    skinned = _skinned(obj, rig)
    missing = unbound(obj, rig) if skinned else []
    if missing:
        raise ValueError("%s binds joints this rig has not got: %s"
                         % (obj.name, missing))
    detach(obj)
    if skinned:
        mod = obj.modifiers.new("EFB Tool", "ARMATURE")
        mod.object = rig
        obj.matrix_world = rig.matrix_world @ _rest(obj)
        obj[SOCKET_PROP] = bone
        return "skinned"

    swap = _swap(obj, rig, bone) if obj.get(JOINT_PROP) not in (None, bone) else None
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    ctx.view_layer.update()
    if not _from_library(obj):
        obj.matrix_world = rig.matrix_world @ rig.pose.bones[bone].matrix
        kind = "origin"
    else:
        obj.matrix_world = (rig.matrix_world @ deform_matrix(rig, bone)
                            @ (swap if swap is not None else IDENTITY) @ _rest(obj))
        kind = "swapped" if swap is not None else "rigid"
    obj[SOCKET_PROP] = bone
    return kind


class _ToolOperator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)


class EFB_OT_attach_tool(_ToolOperator):
    """Hang the chosen object off this socket, so the weapon can be posed with the rig"""
    bl_idname = "efb.attach_tool"
    bl_label = "Attach to Socket"

    socket: StringProperty(default="Tool_R")

    def execute(self, context):
        rig = context.object
        s = getattr(rig, "efb_rig", None)
        obj = getattr(s, "tool_object", None) if s else None
        if obj is None:
            self.report({"ERROR"}, "pick an object to attach first")
            return {"CANCELLED"}
        try:
            kind = attach(rig, obj, self.socket or s.tool_socket, context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "%s on %s (%s)" % (obj.name, self.socket, kind))
        return {"FINISHED"}


class EFB_OT_detach_tool(_ToolOperator):
    """Take the chosen object back off its socket, leaving it where it is"""
    bl_idname = "efb.detach_tool"
    bl_label = "Detach from Socket"

    def execute(self, context):
        s = getattr(context.object, "efb_rig", None)
        obj = getattr(s, "tool_object", None) if s else None
        if obj is None:
            self.report({"ERROR"}, "pick an object first")
            return {"CANCELLED"}
        detach(obj)
        self.report({"INFO"}, "%s detached" % obj.name)
        return {"FINISHED"}


def draw_tool_row(layout, rig):
    """The weapon preview control. Shared by the sidebar and the Item tab."""
    s = getattr(rig, "efb_rig", None)
    if s is None:
        return
    layout.prop(s, "show_tool_proxy", toggle=True, icon="EMPTY_SINGLE_ARROW")
    col = layout.column(align=True)
    col.scale_y = 0.8
    col.label(text="The stick points where the weapon points.", icon="INFO")
    col = layout.column(align=True)
    col.prop(s, "tool_object", text="")
    obj = s.tool_object
    if obj is None:
        col.label(text="Pick an object to hang on a socket", icon="INFO")
        return
    holder, bone = attached_to(obj)
    row = col.row(align=True)
    row.prop(s, "tool_socket", text="")
    op = row.operator("efb.attach_tool", text="Attach", icon="LINKED")
    op.socket = s.tool_socket
    if holder is rig and bone:
        row = col.row(align=True)
        row.label(text="on %s" % bone, icon="CHECKMARK")
        row.operator("efb.detach_tool", text="Detach", icon="UNLINKED")


CLASSES = (EFB_OT_attach_tool, EFB_OT_detach_tool)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
