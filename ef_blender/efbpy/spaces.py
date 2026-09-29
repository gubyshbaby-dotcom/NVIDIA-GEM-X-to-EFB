"""Which space the pole rides: the limb's own chord, or the root.

The reference rig's poles stand still while the limb swings, and nothing moves them but the
animator. That is one of the two settings here. The other is the chord - the limb's
root-to-handle line - which folds 0.82 percent of 4518 shipped poses to the wrong side
against the standing pole's 29.37 percent, so it is the default and the panel says both
numbers.

Standing still is against the master control, not against armature space. The reference
leaves its poles loose and drives away from them with the root; that is the one part of its
arrangement we do not copy, because the root has to carry the whole figure.

The switch itself is two Child Ofs on a holder driven complementary, by one rig property.
All that lives here is holding the control's world pose across the change, which is what
makes it usable mid-shot: the pole does not jump when its space does.
"""

import bpy
from bpy.props import BoolProperty, StringProperty

from efb.rigdef import POLE_FOLLOW_MEASURED

from . import snap
from .ops import rig_limb
from .restpose import set_matrix

LABELS = ("Root space", "Chord")


def pole_follow_prop(limb):
    return getattr(limb, "pole_follow", "")


def follows_chord(rig, limb) -> bool:
    return bool(rig.get(pole_follow_prop(limb), 0))


def set_pole_follow(rig, limb, on, context=None, keep_place=True):
    """Move this limb's pole between the chord and the root.

    Exact rather than approximate, because the Child Of is on the holder and the control
    is an ordinary child of it. A rest override is the one constraint the control does
    carry itself, so the write goes through set_matrix, which takes it back out.
    """
    context = context or bpy.context
    prop = pole_follow_prop(limb)
    if not prop:
        return
    pb = rig.pose.bones[limb.pole_ctrl]
    before = pb.matrix.copy() if keep_place else None
    rig[prop] = 1.0 if on else 0.0
    rig.update_tag()
    context.view_layer.update()
    if before is not None:
        set_matrix(pb, before)
        context.view_layer.update()


class EFB_OT_set_pole_follow(bpy.types.Operator):
    """Ride this pole on the limb's own chord, or leave it standing against the root the
    way the reference rig's poles stand still. The chord folds far more shipped poses to
    the side Epic Fight's own clips fold to"""
    bl_idname = "efb.set_pole_follow"
    bl_label = "Set Pole Space"
    bl_options = {"REGISTER", "UNDO"}

    limb: StringProperty(name="Limb", default="arm_r")
    follow: BoolProperty(name="Follow the chord", default=True)

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        limb = rig_limb(rig, self.limb)
        if not pole_follow_prop(limb):
            self.report({"ERROR"}, "%s has no pole space switch" % limb.label)
            return {"CANCELLED"}
        set_pole_follow(rig, limb, self.follow, context)
        self.report({"INFO"}, "%s pole now in %s"
                    % (limb.label, LABELS[1 if self.follow else 0]))
        return {"FINISHED"}


def draw_pole_space(layout, rig, limb):
    """Two lit buttons and the measurement under them. The number is the point: an
    animator asked to match the reference here is giving up a nineteen-fold accuracy gain
    and has a right to see the size of it before deciding."""
    if not pole_follow_prop(limb):
        return
    now = follows_chord(rig, limb)
    row = layout.row(align=True)
    for i, label in enumerate(LABELS):
        op = row.operator("efb.set_pole_follow", text=label, depress=(bool(i) == now))
        op.limb, op.follow = limb.key, bool(i)
    note = layout.column(align=True)
    note.scale_y = 0.8
    note.label(text="Chord %.2f%% wrong-side folds, root space %.2f%%,"
                    % (POLE_FOLLOW_MEASURED["chord"], POLE_FOLLOW_MEASURED["world"]),
               icon="INFO")
    note.label(text="over %d shipped poses. Root space is the"
                    % POLE_FOLLOW_MEASURED["poses"])
    note.label(text="reference's standing pole, carried by the root.")


CLASSES = (EFB_OT_set_pole_follow,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
