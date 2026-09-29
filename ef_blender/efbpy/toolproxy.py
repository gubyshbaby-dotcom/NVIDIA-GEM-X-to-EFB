"""A stick down each weapon socket, so which way the tool points reads from any angle.

The socket widget is a wire cage inside a fist and it is capped at 0.33 m by the controls
around it - the IK hand cube owns the grip, the elbow pole owns the air past 0.34 m. A
proxy is not a bone widget, so none of that applies to it: it is an ordinary mesh object
on the socket and it can be as long as the weapon it stands for.

It is placed by efbpy.tool.attach, the same call the weapon preview uses, so the stick and
a real weapon land on the same matrix. Nothing here is exported: the exporter reads the
rig's bones and a child mesh is not one.
"""

import bpy
from bpy.props import StringProperty

from efb.rigedit import TOOL_AXIS, TOOL_AXIS_SIGN, TOOL_SOCKETS

from . import snap, tool

__all__ = ["ensure", "clear", "proxies", "PROXY_PROP", "LENGTH"]

PROXY_PROP = "efb_tool_proxy"

LENGTH = 0.55
_SHAFT = 0.72
_HALF = 0.022


def _bar():
    """A tapered bar down the socket's tool axis, as (verts, faces).

    Nothing sits behind the socket, not even a pommel: an 11:1 stick still has two ends to
    compare before it says anything, and from a foreshortened angle the short end wins.
    """
    a, b, c = 0.0, LENGTH * _SHAFT, LENGTH
    h = _HALF
    verts = [(-h, -h, a), (h, -h, a), (h, h, a), (-h, h, a),
             (-h, -h, b), (h, -h, b), (h, h, b), (-h, h, b),
             (0.0, 0.0, c)]
    faces = [(3, 2, 1, 0), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
             (4, 8, 5), (5, 8, 6), (6, 8, 7), (7, 8, 4)]
    if TOOL_AXIS != 2 or TOOL_AXIS_SIGN < 0.0:
        verts = [_on_axis(v) for v in verts]
        faces = [tuple(reversed(f)) for f in faces]
    return verts, faces


def _on_axis(v):
    """A vertex built along +Z, moved onto the socket's real tool axis."""
    rest = [c for i, c in enumerate(v) if i != 2]
    out = list(rest)
    out.insert(TOOL_AXIS, v[2] * TOOL_AXIS_SIGN)
    return tuple(out)


def proxies(rig):
    """Every stick currently on this rig, by socket."""
    out = {}
    for obj in bpy.data.objects:
        socket = obj.get(PROXY_PROP)
        if socket and obj.parent is rig:
            out[socket] = obj
    return out


def _make(rig, socket, context):
    verts, faces = _bar()
    mesh = bpy.data.meshes.new("EFB-Tool-Proxy-" + socket)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("EFB-Tool-Proxy-" + socket, mesh)
    for coll in (rig.users_collection or (context.scene.collection,)):
        coll.objects.link(obj)
    obj[PROXY_PROP] = socket
    obj.hide_render = True
    obj.hide_select = True
    obj.display_type = "SOLID"
    tool.attach(rig, obj, socket, context)
    return obj


def ensure(rig, context=None):
    """A stick on every socket this rig has. Idempotent."""
    context = context or bpy.context
    have = proxies(rig)
    made = []
    for socket in TOOL_SOCKETS:
        if socket in have or socket not in rig.pose.bones:
            continue
        made.append(_make(rig, socket, context))
    return made


def clear(rig):
    """Take every stick off, mesh and all."""
    gone = 0
    for obj in list(proxies(rig).values()):
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        gone += 1
    return gone


def apply_proxy(rig, context=None):
    """Bring the sticks into line with the rig's own switch. The update callback."""
    settings = getattr(rig, "efb_rig", None)
    if settings is None:
        return
    if settings.show_tool_proxy:
        ensure(rig, context)
    else:
        clear(rig)


class _ProxyOperator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)


class EFB_OT_tool_proxy(_ProxyOperator):
    """Put a stick down each weapon socket, pointing the way the tool points. It is a
    viewport marker: it does not render and it is not exported"""
    bl_idname = "efb.tool_proxy"
    bl_label = "Show Tool Direction"

    mode: StringProperty(default="TOGGLE")

    def execute(self, context):
        rig = context.object
        settings = getattr(rig, "efb_rig", None)
        if settings is None:
            self.report({"ERROR"}, "this rig has no add-on settings yet - press Apply Rig")
            return {"CANCELLED"}
        on = not settings.show_tool_proxy if self.mode == "TOGGLE" else self.mode == "ON"
        settings.show_tool_proxy = on
        self.report({"INFO"}, "tool direction %s" % ("shown" if on else "hidden"))
        return {"FINISHED"}


CLASSES = (EFB_OT_tool_proxy,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
