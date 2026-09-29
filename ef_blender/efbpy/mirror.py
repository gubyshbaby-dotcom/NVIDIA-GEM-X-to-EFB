"""Mirror a pose across the body's own plane.

Measured before it was written: the Epic Fight rest is an X mirror of itself to 2.2e-05
(the float32 floor of biped.json), and Blender's own flip-side naming turns all 40 sided
control names into their partners with no misses. Both facts are what make the mirror a
one-line conjugation instead of a table.

    basis_L = S @ basis_R @ S,   S = diag(-1, 1, 1)

That follows from rest_L = S rest_R S by induction down the chain, so it needs no
view-layer update between bones and no parent ordering. It is done in basis space, not
world space, for exactly that reason.
"""

import bpy
from bpy.props import BoolProperty
from mathutils import Matrix

from . import snap
from .ops import rig_limbs
from .restpose import is_override, local_offset

S = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))

_LIMB_PROPS = ("prop", "end_lock", "pole_follow")


def partner(rig, name):
    """The bone on the other side, or None. Blender's flip covers every control; the
    hidden mechanism bones carry their side in the middle of the name, so they get the
    plain swap - MCH-POLE-Elbow_R.chord flips to the left arm's chord."""
    flipped = bpy.utils.flip_name(name)
    if flipped != name and flipped in rig.pose.bones:
        return flipped
    for a, b in (("_R", "_L"), ("_L", "_R")):
        if a in name:
            other = name.replace(a, b, 1)
            if other in rig.pose.bones:
                return other
    return None


def mirror_pose(rig, bones=None, context=None):
    """Mirror the given bones onto their partners, and the centre ones onto themselves.
    Returns how many bones were written. Reads everything before writing anything, so a
    left/right pair swaps rather than one side overwriting the other."""
    context = context or bpy.context
    names = list(bones) if bones is not None else [pb.name for pb in rig.pose.bones]
    source = {}
    for name in names:
        pb = rig.pose.bones.get(name)
        if pb is None or is_override(pb):
            continue
        target = partner(rig, name) or name
        source[target] = S @ _visible(pb) @ S
    for target, local in source.items():
        pb = rig.pose.bones[target]
        off = local_offset(pb)
        pb.matrix_basis = local if off is None else off.inverted() @ local
    _swap_limb_props(rig, set(source))
    rig.update_tag()
    context.view_layer.update()
    return len(source)


def _visible(pose_bone):
    """What the animator sees in this bone's local space: its own pose composed onto
    whatever its rest override is holding. Mirroring the basis alone flips only half of a
    control that carries one."""
    off = local_offset(pose_bone)
    basis = pose_bone.matrix_basis.copy()
    return basis if off is None else off @ basis


def _swap_limb_props(rig, written):
    """Swap the switches of any limb pair the mirror actually touched."""
    limbs = {l.key: l for l in rig_limbs(rig)}
    for right, left in (("arm_r", "arm_l"), ("leg_r", "leg_l")):
        a, b = limbs.get(right), limbs.get(left)
        if a is None or b is None:
            continue
        if not (_limb_touched(a, written) or _limb_touched(b, written)):
            continue
        for attr in _LIMB_PROPS:
            pa, pb_ = getattr(a, attr, ""), getattr(b, attr, "")
            if pa in rig and pb_ in rig:
                rig[pa], rig[pb_] = rig[pb_], rig[pa]


def _limb_touched(limb, written):
    return any(getattr(limb, a) in written
               for a in ("fk_upper", "fk_lower", "ik_ctrl", "pole_ctrl"))


def selected_controls(rig):
    """The bones the animator has picked. Both flags are measured, not assumed: Bone has
    no `select` at all, and pose-mode H sets PoseBone.hide while the rig's own structural
    hide is on Bone.hide - two different flags, so a bone hidden either way is skipped."""
    return [pb.name for pb in rig.pose.bones
            if pb.select and not (pb.hide or pb.bone.hide)]


class EFB_OT_mirror_pose(bpy.types.Operator):
    """Flip the pose left to right. With bones selected only those are mirrored, with
    nothing selected the whole rig is"""
    bl_idname = "efb.mirror_pose"
    bl_label = "Mirror Pose"
    bl_options = {"REGISTER", "UNDO"}

    selected_only: BoolProperty(
        name="Selected only", default=True,
        description="Off, the whole rig is mirrored whatever is selected")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object) and context.mode == "POSE"

    def execute(self, context):
        rig = context.object
        chosen = selected_controls(rig) if self.selected_only else None
        n = mirror_pose(rig, chosen or None, context)
        self.report({"INFO"}, "mirrored %d bones" % n)
        return {"FINISHED"}


CLASSES = (EFB_OT_mirror_pose,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
