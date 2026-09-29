"""Wide (Steve) <-> Slim (Alex) on a rig that already exists, in place.

The two builds are one rig. Measured over the whole definition, they differ in nothing
but the rest matrices of 42 arm-side bones - the shoulders, the arm chains and their
controls, poles, sockets and mechanism, about 3 cm inward on the slim build. Every
constraint, driver, property, pole angle and bone name is the same. So a switch does not
rebuild anything: it moves those rests, re-seats the Child Ofs whose inverse is a rest,
and swaps the body for the other build's.

Everything an animator made survives it. Epic Fight keys are deltas from the rest - that
is what the game stores and why one clip plays on both builds - so every action, the one
playing and the ones parked on NLA tracks, keeps its keys and plays on the new arms the way
the game would play it. IK handle keys are deltas from the handle's rest too, so a limb
posed on IK keeps its arm shape. What does not carry over is an IK bake made off FK - the
handle rests moved - so the playing action's inert bakes are made again. Rest overrides,
attached weapons, settings and the rig object itself are untouched.

The skin follows the build when it is the bundled one (Steve for wide, Alex for slim); a
skin of the animator's own is kept, because only they know which build it was drawn for.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty

from efb import bundled
from efb.rigdef import RIG_ID, RIG_VERSION, build_biped_rig, validate

from . import body, smooth, snap
from .animscene import ARMATURE_PROP, to_matrix
from .ops import BUILDS, RIG_NAMES, named_rig
from .restpose import OF

__all__ = ["rig_build", "switch_build", "EFB_OT_switch_build", "draw_build_switch",
           "register", "unregister"]


def rig_build(rig):
    """"biped" or "biped_slim_arm", or None for a rig no bundled build made."""
    v = rig.get(ARMATURE_PROP) if rig is not None else None
    return v if v in bundled.VARIANTS else None


def _rests(variant):
    arm = bundled.armature(variant)
    rigdef = build_biped_rig(arm)
    errors = validate(rigdef)
    if errors:
        raise RuntimeError("the %s definition is invalid: %s" % (variant, errors[0]))
    return rigdef, {b.name: b.rest for b in rigdef.bones}


def _set_rests(rig, rests, context):
    """Edit-bone matrices onto `rests`; a rest override's duplicate follows its control.
    Returns (bones moved, bones this rig has that the definition does not)."""
    targets, foreign = {}, []
    for bone in rig.data.bones:
        key = bone.name if bone.name in rests else bone.get(OF)
        if key in rests:
            targets[bone.name] = to_matrix(rests[key])
        else:
            foreign.append(bone.name)

    view = context.view_layer
    prev_active = view.objects.active
    prev_mode = rig.mode
    hidden = rig.hide_get()
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    rig.hide_set(False)
    view.objects.active = rig
    moved = 0
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for eb in rig.data.edit_bones:
            want = targets.get(eb.name)
            if want is None:
                continue
            got = eb.matrix
            if any(abs(got[r][c] - want[r][c]) > 1e-6 for r in range(4) for c in range(4)):
                eb.matrix = want
                moved += 1
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        if prev_mode != "OBJECT" and prev_active is rig:
            bpy.ops.object.mode_set(mode=prev_mode)
        view.objects.active = prev_active
        rig.hide_set(hidden)
    return moved, foreign


def _reseat(rig, rigdef):
    """A seated Child Of holds the inverse of its target's rest - measured on a fresh rig
    to 5e-7 - so it is set from the new rest directly, whatever the rig is posing."""
    count = 0
    for c in rigdef.constraints:
        if not c.seat_inverse or c.owner not in rig.pose.bones:
            continue
        con = rig.pose.bones[c.owner].constraints.get(c.name)
        if con is None or c.subtarget not in rig.data.bones:
            continue
        con.inverse_matrix = rig.data.bones[c.subtarget].matrix_local.inverted()
        count += 1
    return count


def _bundled_skin(image):
    path = getattr(image, "filepath", "") if image else ""
    return not path or any(body._same_file(path, bundled.skin_path(v))
                           for v in bundled.VARIANTS)


def _swap_bodies(rig, variant, keep_custom_skin, context):
    """Replace both bodies with the other build's. Returns (rebuilt, skin kept)."""
    old = body.bodies_of(rig)
    if not old:
        return False, False
    game = smooth.body_of(rig)
    material = game.data.materials[0] if game and game.data.materials else None
    keep = keep_custom_skin and material is not None \
        and not _bundled_skin(body.skin_image_of(material))
    collections = list(game.users_collection) if game else []
    hidden = {o.get("efb_body") is not None: o.hide_get() for o in old}
    for obj in old:
        me = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)

    mesh = bundled.body_mesh(variant)
    new = body.build_body(mesh, rig, skin_path="" if keep else None, variant=variant,
                          context=context)
    if keep:
        new.data.materials.append(material)
    authoring = smooth.build_smooth_body(new, rig, context=context)
    for obj in (new, authoring):
        if collections:
            for coll in list(obj.users_collection):
                coll.objects.unlink(obj)
            for coll in collections:
                coll.objects.link(obj)
    new.hide_set(hidden.get(True, False))
    authoring.hide_set(hidden.get(False, False))
    return True, keep


def _rebake(rig, context):
    """The playing action's IK bakes that no frame uses: the handle rests moved under
    them, so they are made again off the FK the action still plays."""
    from .animscene import on_ik
    from .ikbake import bake_frames, bake_ik, baked_limbs

    ad = rig.animation_data
    action = ad.action if ad else None
    if action is None:
        return []
    limbs = [l for l in baked_limbs(rig, action) if not on_ik(rig, l, action)]
    if not limbs:
        return []
    start, end = (int(round(v)) for v in action.frame_range)
    bake_ik(rig, bake_frames((), start, max(end, start + 1)), limbs=limbs, context=context)
    return [l.key for l in limbs]


def switch_build(rig, variant, context=None, keep_custom_skin=True) -> dict:
    """Move `rig` onto another bundled build. Returns what was done, for the report."""
    ctx = context or bpy.context
    old = rig_build(rig)
    if old is None:
        raise RuntimeError("%s was not generated from a bundled build" % rig.name)
    if variant not in bundled.VARIANTS:
        raise RuntimeError("no bundled build %r" % variant)
    if old == variant:
        return {"changed": False}
    if rig.data.get("rig_id") == RIG_ID and rig.data.get("rig_version", RIG_VERSION) \
            != RIG_VERSION:
        raise RuntimeError("%s is a version %s rig; migrate it to version %d first"
                           % (rig.name, rig.data.get("rig_version"), RIG_VERSION))

    rigdef, rests = _rests(variant)
    moved, foreign = _set_rests(rig, rests, ctx)
    seated = _reseat(rig, rigdef)
    rig[ARMATURE_PROP] = variant
    if rig.name == RIG_NAMES.get(old):
        rig.name = RIG_NAMES[variant]
        rig.data.name = RIG_NAMES[variant]
    rig.update_tag()
    ctx.view_layer.update()
    rebuilt, kept = _swap_bodies(rig, variant, keep_custom_skin, ctx)
    rebaked = _rebake(rig, ctx)
    ctx.view_layer.update()
    return {"changed": True, "from": old, "to": variant, "moved": moved,
            "seated": seated, "foreign": foreign, "bodies": rebuilt, "skin_kept": kept,
            "rebaked": rebaked}


class EFB_OT_switch_build(bpy.types.Operator):
    """Switch this rig between the wide (Steve, 4px arms) and slim (Alex, 3px arms)
    builds in place. Keys, parked clips, settings, overrides and attached weapons are
    kept; the body is swapped"""
    bl_idname = "efb.switch_build"
    bl_label = "Switch Build"
    bl_options = {"REGISTER", "UNDO"}

    variant: EnumProperty(name="Build", items=BUILDS, default=bundled.DEFAULT_VARIANT)
    rig: StringProperty(name="Rig", default="",
                        description="Which rig to switch. Blank uses the active one")
    keep_custom_skin: BoolProperty(
        name="Keep my skin", default=True,
        description="A skin of your own stays on the new body. The bundled Steve and Alex "
                    "skins always follow the build")

    @classmethod
    def poll(cls, context):
        return rig_build(context.object) is not None and snap.is_rig(context.object)

    def execute(self, context):
        rig = named_rig(self, context)
        if rig is None or rig_build(rig) is None:
            self.report({"ERROR"}, "not a rig generated from a bundled build")
            return {"CANCELLED"}
        try:
            done = switch_build(rig, self.variant, context, self.keep_custom_skin)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not done["changed"]:
            self.report({"INFO"}, "%s is already %s" % (rig.name, _label(self.variant)))
            return {"FINISHED"}
        parts = ["%s -> %s" % (_label(done["from"]), _label(done["to"])),
                 "%d bones moved" % done["moved"]]
        if done["bodies"]:
            parts.append("body rebuilt, %s" % ("your skin kept" if done["skin_kept"]
                                               else "bundled skin"))
        if done["rebaked"]:
            parts.append("IK re-baked on %s" % ", ".join(done["rebaked"]))
        if done["foreign"]:
            parts.append("%d bones of your own left in place" % len(done["foreign"]))
        self.report({"INFO"}, "%s: %s" % (rig.name, "; ".join(parts)))
        return {"FINISHED"}


def _label(variant):
    return next((label for key, label, _d in BUILDS if key == variant), variant)


def draw_build_switch(layout, rig):
    """Two lit buttons, like the FK/IK pair: the build the rig is, and the other one."""
    now = rig_build(rig)
    if now is None:
        return
    row = layout.row(align=True)
    row.label(text="Build", icon="OUTLINER_OB_ARMATURE")
    for key, label, _desc in BUILDS:
        op = row.operator("efb.switch_build", text=label, depress=(key == now))
        op.variant, op.rig = key, rig.name


CLASSES = (EFB_OT_switch_build,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
