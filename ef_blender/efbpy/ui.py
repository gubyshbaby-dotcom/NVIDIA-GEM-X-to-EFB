"""N-panel. Two halves.

The top half is what changes during a shot: one box per limb with the IK/FK blend, the
snap buttons, and - the reason this file was rewritten - a Find IK Control button, because
arms default to FK and an animator who has never switched one has never seen an arm IK
handle. The same box rides the sidebar's Item tab, keyed to the selected bone.

The bottom half edits the rig itself: widget sizes, joint limits, the seam slide, the
weapon preview and what is drawn. Everything there writes through efbpy.rigedit, which
only ever touches pose-level data, so an animator can change a widget size mid-shot and
keep their action.
"""

import bpy

from efb.rigdef import (GLOBAL_HEAD_ROTATION, LOCK_LIMB_ROTATION, SHOW_INACTIVE,
                        SMOOTH_DEFORM, TOOL_JOINTS, tool_bind_prop, tool_keyed_prop)
from efb.rigedit import AXIS_GROUPS, COLLECTION_LABELS, WIDGET_GROUPS, group_of

from . import body, migrate, restpose, rigedit, rootcarry, snap, spaces, tool
from .ops import limb_of, rig_limbs

CATEGORY = "Epic Fight"


class _RigPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)


def _state(value):
    """What the blend reads as, for the box header."""
    if value >= 0.999:
        return "IK", "CON_KINEMATIC"
    if value <= 0.001:
        return "FK", "BONE_DATA"
    return "IK %d%%" % round(value * 100.0), "CON_ROTLIKE"


def _ready(layout, obj):
    """The settings block, or a one-click setup for a rig that predates it."""
    s = rigedit.settings(obj)
    if s is None or not s.initialised or not len(s.limbs):
        layout.operator("efb.rig_setup", icon="FILE_REFRESH")
        return None
    return s


def is_advanced(obj):
    """Advanced only ever ADDS rows. The reference's own panel drops two toggles when it
    is turned on, which is the one thing not to copy."""
    s = rigedit.settings(obj)
    return bool(s is not None and s.advanced)


class EFB_PT_rig(_RigPanel):
    bl_label = "Epic Fight Rig"
    bl_idname = "EFB_PT_rig"

    def draw_header(self, context):
        self.layout.label(text="", icon="ARMATURE_DATA")

    def draw(self, context):
        obj = context.object
        layout = self.layout
        row = layout.row()
        row.label(text=obj.name, icon="OUTLINER_OB_ARMATURE")
        row.label(text="v%s" % obj.data.get("rig_version", "?"))
        row = layout.row(align=True)
        row.operator("efb.reset_pose", icon="LOOP_BACK")
        if "snap_keyframe" in obj:
            row.prop(obj, '["snap_keyframe"]', text="Key on Snap", toggle=True,
                     icon="DECORATE_KEYFRAME")
        if SHOW_INACTIVE in obj:
            layout.prop(obj, '["%s"]' % SHOW_INACTIVE, text="Show FK and IK together",
                        toggle=True, icon="HIDE_OFF")
        s = rigedit.settings(obj)
        if s is not None:
            layout.prop(s, "advanced", toggle=True,
                        icon="PREFERENCES" if s.advanced else "DOT")


def _limb_box(layout, obj, limb, advanced=False):
    """FK and IK as two lit buttons rather than a number an animator has to read. The
    Find button is the answer to "the rig has no arm IK": it switches the limb and puts
    the handle under the cursor."""
    value = obj[limb.prop]
    box = layout.box()
    col = box.column(align=True)
    text, icon = _state(value)
    row = col.row(align=True)
    row.label(text=limb.label, icon=icon)
    row.label(text=text)

    row = col.row(align=True)
    op = row.operator("efb.set_limb_mode", text="FK", depress=value <= 0.001)
    op.limb, op.mode = limb.key, "FK"
    op = row.operator("efb.set_limb_mode", text="IK", depress=value >= 0.999)
    op.limb, op.mode = limb.key, "IK"

    col.prop(obj, '["%s"]' % limb.prop, text="Blend", slider=True)
    _joint_row(col, obj, limb, value)
    row = col.row(align=True)
    op = row.operator("efb.snap_ik_to_fk", text="IK <- FK", icon="SNAP_ON")
    op.limb = limb.key
    op = row.operator("efb.snap_fk_to_ik", text="FK <- IK", icon="SNAP_ON")
    op.limb = limb.key
    if value < 0.999:
        row = col.row(align=True)
        row.alert = value <= 0.001
        row.operator("efb.reveal_ik", icon="VIEWZOOM").limb = limb.key
    if advanced:
        _limb_advanced(box, obj, limb, value)


def _joint_row(layout, obj, limb, value):
    """What turns the lower segment, and where its roll comes from.

    The elbow / knee carries one control now, the FK ring, and the seam marker carries
    none - which is the reference rig exactly. Under IK the segment is the solve's, so a
    control at the joint could only lie about it and the panel has to say where the real
    one is.
    """
    lower = "forearm and hand" if limb.key.startswith("arm") else "shin and foot"
    end = "wrist" if limb.key.startswith("arm") else "ankle"
    joint = "Elbow" if limb.key.startswith("arm") else "Knee"
    col = layout.column(align=True)
    col.scale_y = 0.8
    if value <= 0.001:
        col.label(text="%s turns the %s." % (limb.fk_lower, lower), icon="INFO")
        return
    col.label(text="Under IK the %s is solved." % lower.split(" and ")[0], icon="INFO")
    if rigedit.pinned(obj, limb.key):
        col.label(text="Nothing rolls it: Fold on one axis only holds")
        col.label(text="its roll at zero. Joint Limits has the switch.")
        return
    col.label(text="%s rolls it; the %s does not turn it."
              % (limb.ik_ctrl, joint.lower()))
    lock = getattr(limb, "end_lock", "")
    if lock in obj and float(obj[lock]) <= 0.001:
        col.label(text="%s Lock is 0, so the solve keeps the roll." % end.capitalize(),
                  icon="ERROR")


def _limb_advanced(box, obj, limb, value):
    """The rows an animator reaches for a few times a shot rather than every minute. Both
    are inert under FK, so the block is dimmed there instead of hidden - a row that
    vanishes is a row nobody finds twice."""
    col = box.column(align=True)
    live = value > 0.001
    col.active = live
    end = "Wrist Lock" if limb.key.startswith("arm") else "Ankle Lock"
    held = rigedit.pinned(obj, limb.key)
    lock = getattr(limb, "end_lock", "")
    if lock in obj:
        row = col.row(align=True)
        row.active = live and not held
        row.prop(obj, '["%s"]' % lock, text=end, slider=True)
    if held:
        note = col.column(align=True)
        note.scale_y = 0.8
        note.label(text="%s is inert: the limb folds on one axis only." % end,
                   icon="LOCKED")
    if getattr(limb, "pole_follow", "") in obj:
        col.label(text="Pole Space", icon="CON_TRACKTO")
        spaces.draw_pole_space(col, obj, limb)
    _ownership_note(box, obj, limb, value)


def _ownership_note(box, obj, limb, value):
    """What the joint row does not say: the upper segment is untouched by the lower one
    under FK, and under IK the fold is a solved number rather than a dial nobody built."""
    col = box.column(align=True)
    col.scale_y = 0.8
    joint = "Elbow" if limb.key.startswith("arm") else "Knee"
    if value <= 0.001:
        col.label(text="The upper segment does not move with it.", icon="INFO")
        return
    col.label(text="The handle fixes the %s: under IK its" % joint.lower(), icon="INFO")
    col.label(text="fold is one number and no dial can change it.")


class _LimbPanel(_RigPanel):
    bl_parent_id = "EFB_PT_rig"
    prefix = ""

    def draw(self, context):
        obj = context.object
        row = self.layout.row()
        right = row.column(align=True)
        left = row.column(align=True)
        advanced = is_advanced(obj)
        mine = [l for l in rig_limbs(obj)
                if l.key.startswith(self.prefix) and l.prop in obj]
        for limb in mine:
            _limb_box(right if limb.key.endswith("_r") else left, obj, limb, advanced)
        loose = [l.label for l in mine if not obj.get(getattr(l, "pole_follow", ""), 1)]
        if loose:
            self.layout.label(
                text="%s: pole in armature space, 29.37%% wrong-side folds"
                     % ", ".join(loose), icon="INFO")


class EFB_PT_arms(_LimbPanel):
    bl_label = "Arms"
    bl_idname = "EFB_PT_arms"
    prefix = "arm"


class EFB_PT_legs(_LimbPanel):
    bl_label = "Legs"
    bl_idname = "EFB_PT_legs"
    prefix = "leg"


class EFB_PT_item(bpy.types.Panel):
    """The blend for the limb the selected bone belongs to, in the tab that is already
    open. An IK control the animator can see is live, but this says so in words."""
    bl_label = "Epic Fight"
    bl_idname = "EFB_PT_item"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object) and context.mode == "POSE"

    def draw(self, context):
        obj = context.object
        bone = context.active_pose_bone
        limb = limb_of(obj, bone.name) if bone else None
        if limb is not None and limb.prop in obj:
            _limb_box(self.layout, obj, limb, is_advanced(obj))
        elif bone is None or not bone.name.startswith("CTRL-Tool"):
            self.layout.label(text="No IK limb selected", icon="INFO")
        joint = bone.name[len("CTRL-"):] if bone else ""
        if joint in TOOL_JOINTS and tool_bind_prop(joint) in obj:
            box = self.layout.box()
            box.label(text="Tool Bind", icon="TOOL_SETTINGS")
            _socket_row(box, obj, joint)
        box = self.layout.box()
        box.label(text="Weapon Preview", icon="TOOL_SETTINGS")
        tool.draw_tool_row(box, obj)


class EFB_PT_body(_RigPanel):
    """The switches that are not per-limb, plus the two pose-wide tools. Everything on
    this panel writes a property on the rig object, so it keys and it still works with the
    addon disabled - except Mirror and the rest pose, which are operators by nature."""
    bl_label = "Body"
    bl_idname = "EFB_PT_body"
    bl_parent_id = "EFB_PT_rig"

    def draw(self, context):
        obj = context.object
        layout = self.layout
        row = layout.row(align=True)
        for prop, label, icon in ((GLOBAL_HEAD_ROTATION, "Global Head", "ORIENTATION_GLOBAL"),
                                  (LOCK_LIMB_ROTATION, "Lock Limbs", "CON_ROTLIKE")):
            if prop in obj:
                row.prop(obj, '["%s"]' % prop, text=label, toggle=True, icon=icon)

        _body_switch(layout, obj)
        _skin(layout, obj)
        _tool_bind(layout, obj)

        row = layout.row(align=True)
        row.operator("efb.mirror_pose", icon="MOD_MIRROR")

        if not is_advanced(obj):
            return
        box = layout.box()
        box.label(text="Rest Pose", icon="ARMATURE_DATA")
        held = len(restpose.overridden(obj))
        row = box.row(align=True)
        row.operator("efb.update_rest_pose", text="Update", icon="FILE_REFRESH")
        sub = row.row(align=True)
        sub.active = held > 0
        sub.operator("efb.reset_rest_pose", text="Reset", icon="LOOP_BACK")
        box.operator("efb.straighten_arms", text="Straighten Arms", icon="CON_KINEMATIC")
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="%d controls hold an override" % held,
                  icon="CHECKMARK" if held else "INFO")
        col.label(text="The 20 joint rest matrices are never moved.")


def _body_switch(layout, obj):
    """Which of the two bodies is drawn.

    The authoring body is a stylised shape, not a preview of the game: measured against
    Epic Fight's own skinning it is 0.043 to 0.196 m out on shipped poses, while the game
    body reproduces it to 1e-6. The panel says which one is on the screen, in those words.
    """
    if SMOOTH_DEFORM not in obj:
        return
    stylised = bool(obj[SMOOTH_DEFORM])
    box = layout.box()
    box.prop(obj, '["%s"]' % SMOOTH_DEFORM, toggle=True, icon="MOD_SMOOTH",
             text="Body: %s" % ("Stylised preview" if stylised else "Game (exact)"))
    col = box.column(align=True)
    col.scale_y = 0.8
    if stylised:
        col.label(text="NOT what the game draws - up to 0.20 m", icon="ERROR")
        col.label(text="off it. The export is unchanged.")
    else:
        col.label(text="Epic Fight's own skinning, to 1e-6 m.", icon="CHECKMARK")


def _skin(layout, obj):
    """Retarget this rig's skin. Per rig on purpose - the button carries the name of the
    rig it was drawn for, so a scene of several reskins one at a time."""
    if not body.bodies_of(obj):
        return
    image = body.skin_of(obj)
    box = layout.box()
    box.label(text="Skin", icon="TEXTURE")
    row = box.row(align=True)
    op = row.operator("efb.set_skin", icon="FILE_IMAGE",
                      text=image.name if image else "Change Skin")
    op.rig, op.default_skin = obj.name, False
    op = row.operator("efb.set_skin", text="", icon="LOOP_BACK")
    op.rig, op.default_skin = obj.name, True
    col = box.column(align=True)
    col.scale_y = 0.8
    col.label(text="Both bodies. The pose and action are kept.")


def _tool_bind(layout, obj):
    """Bind or unbind each Tool socket from the hand, and decide whether that state is a
    scene switch or an animation channel. Unbound is a real channel, not a viewport trick,
    so the warning about key density is the honest half.

    The way back sits in the same row as the way out. Unbinding leaves the weapon wherever
    the animator dragged it; Return walks it onto the placement the hand dictates, rotation
    included, and binds it again.
    """
    present = [j for j in TOOL_JOINTS if tool_bind_prop(j) in obj]
    if not present:
        return
    box = layout.box()
    box.label(text="Tool Bind", icon="TOOL_SETTINGS")
    free, keyed = [], []
    for joint in present:
        bound, on = _socket_row(box, obj, joint)
        if on:
            keyed.append(joint)
        if not bound:
            free.append(joint)
    col = box.column(align=True)
    col.scale_y = 0.8
    col.label(text="Return puts a socket back where the hand", icon="LOOP_BACK")
    col.label(text="holds it, position and rotation, and rebinds")
    col.label(text="it. Keyed off, it moves this frame only.")
    if keyed:
        _bind_keys_note(box, obj, keyed)
    if not free:
        return
    col = box.column(align=True)
    col.scale_y = 0.8
    col.label(text="A free socket writes a counter-animation into", icon="INFO")
    col.label(text="its own track. The export keys it on every")
    col.label(text="frame its parents move on; bake if the parent")
    col.label(text="curve is not linear, or the weapon swims.")


def _socket_row(layout, obj, joint):
    """One socket: bind out, return back, and whether the state is keyed. Returns
    (bound, keyed) so the caller can write the notes that follow from them."""
    bound = bool(obj[tool_bind_prop(joint)])
    row = layout.row(align=True)
    op = row.operator("efb.set_tool_bind", depress=bound,
                      text="%s: %s" % (joint[-1], "Bound" if bound else "Free"),
                      icon="LINKED" if bound else "UNLINKED")
    op.joint, op.bound = joint, not bound
    row.operator("efb.return_tool", text="", icon="LOOP_BACK").joint = joint
    prop = tool_keyed_prop(joint)
    on = False
    if prop in obj:
        on = bool(obj[prop])
        row.prop(obj, '["%s"]' % prop, text="Keyed", toggle=True,
                 icon="DECORATE_KEYFRAME" if on else "DECORATE_ANIMATE")
    return bound, on


def _bind_keys_note(box, obj, keyed):
    """How many switches are in the action, and what one of them does to interpolation.

    Bind is a boolean id property, so no curve can carry a half-bound socket: every
    interpolation still gives a step. What a ramp changes is where the step lands -
    measured, BEZIER and LINEAR both cut at 10.75 for a key at 11, where the interpolated
    value crosses the boolean threshold. CONSTANT is the only one that cuts on the key.
    """
    from .ops import bind_curve
    col = box.column(align=True)
    col.scale_y = 0.8
    for joint in keyed:
        fc = bind_curve(obj, joint)
        col.label(text="%s: %d bind key%s in the action"
                  % (joint, len(fc.keyframe_points) if fc else 0,
                     "" if fc and len(fc.keyframe_points) == 1 else "s"),
                  icon="DECORATE_KEYFRAME")
    col.label(text="A switch is a hard cut: bind is a boolean, so", icon="INFO")
    col.label(text="no curve can half-follow the hand. CONSTANT is")
    col.label(text="written because it is the only one whose step")
    col.label(text="is on the key - a ramp cuts 0.75 frame early.")
    col.label(text="The world-place hold is keyed with it, on both")
    col.label(text="sides, so the keys win over the hold.")


class EFB_PT_visibility(_RigPanel):
    """Anything not used this session should be one click from gone, and the two limb
    collections should still be able to say the words "IK Bones" while they are hidden."""
    bl_label = "Visibility"
    bl_idname = "EFB_PT_visibility"
    bl_parent_id = "EFB_PT_rig"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        obj = context.object
        layout = self.layout
        s = rigedit.settings(obj)
        if s is not None:
            layout.prop(s, "hide_tools", toggle=True,
                        icon="TOOL_SETTINGS" if not s.hide_tools else "CANCEL")

        driven = {c for l in rig_limbs(obj) for c in (l.fk_collection, l.ik_collection)}
        col = layout.column(align=True)
        for c in obj.data.collections_all:
            if c.name in driven:
                continue
            label, icon = COLLECTION_LABELS.get(c.name, (c.name, "GROUP_BONE"))
            col.prop(c, "is_visible", text=label, toggle=True, icon=icon)

        box = layout.box()
        box.label(text="Limb collections follow the blend", icon="DECORATE_DRIVER")
        for limb in rig_limbs(obj):
            row = box.row(align=True)
            row.label(text=limb.label)
            for name, label in ((limb.fk_collection, "FK Bones"),
                                (limb.ik_collection, "IK Bones")):
                coll = obj.data.collections_all.get(name)
                on = bool(coll and coll.is_visible)
                sub = row.row(align=True)
                sub.active = on
                sub.label(text=label, icon="HIDE_OFF" if on else "HIDE_ON")


class EFB_PT_limits(_RigPanel):
    bl_label = "Joint Limits"
    bl_idname = "EFB_PT_limits"
    bl_parent_id = "EFB_PT_rig"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        s = rigedit.settings(context.object)
        if s is not None and s.initialised:
            self.layout.prop(s, "limits", text="")

    def draw(self, context):
        obj = context.object
        layout = self.layout
        s = _ready(layout, obj)
        if s is None:
            return
        layout.active = s.limits
        layout.prop(s, "preset", expand=True)
        for row in s.limbs:
            box = layout.box()
            head = box.row(align=True)
            head.prop(row, "limits", text="")
            head.label(text=row.label)
            body = box.row(align=True)
            body.active = row.limits
            body.prop(row, "hinge_min")
            body.prop(row, "hinge_max")
            if row.limits and not row.hinge_min <= 0.0 <= row.hinge_max:
                warn = box.column(align=True)
                warn.alert = True
                warn.scale_y = 0.8
                warn.label(text="This band excludes the rest pose. Under IK the",
                           icon="ERROR")
                warn.label(text="hand will sit off its handle to hold the stop.")
            pin = box.row(align=True)
            pin.active = row.limits
            pin.prop(row, "pin", toggle=True,
                     icon="LOCKED" if row.pin else "UNLOCKED")
            if row.limits and not row.pin:
                free = box.column(align=True)
                free.scale_y = 0.8
                free.label(text="Unpinned: under IK this fold is held by the", icon="INFO")
                free.label(text="solver's chain only, not by the exported joint.")
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="Fold on one axis only holds the twist and the sideways",
                  icon="INFO")
        col.label(text="fold at zero. The seam between the two boxes of a limb")
        col.label(text="only closes on the fold axis, so anything off it tears")
        col.label(text="the limb open. The reference rig pins the same two.")

        if s.replay:
            box = layout.box()
            box.alert = True
            box.prop(s, "replay", toggle=True, icon="SEQUENCE")
            note = box.column(align=True)
            note.scale_y = 0.8
            note.label(text="An imported clip rolls a forearm or a shin, so the")
            note.label(text="pin is standing down and the clip keeps its roll.")
            note.label(text="The fold is still held - by the FK control and by")
            note.label(text="the solver - but the exported joint carries that")
            note.label(text="roll, so its own numbers can read past the band.")
            note.label(text="Turn this off to author against the pin.")

        box = layout.box()
        box.label(text="Everything else", icon="CON_ROTLIKE")
        for key, (label, icon, _on) in AXIS_GROUPS.items():
            row = box.row(align=True)
            row.prop(s, "axis_" + key, text=label, icon=icon, toggle=True)
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="Epic Fight ranges clamp nothing upstream ships.", icon="INFO")
        col.label(text="A fold band goes on the FK control, on the solver's")
        col.label(text="chain and twice on the Epic Fight joint - before the")
        col.label(text="solve and after it - so the number above is the one")
        col.label(text="that acts on every route and in the file. The upper")
        col.label(text="arm is never limited: shipped clips take it through")
        col.label(text="gimbal in every euler order.")


class EFB_PT_widgets(_RigPanel):
    bl_label = "Control Size"
    bl_idname = "EFB_PT_widgets"
    bl_parent_id = "EFB_PT_rig"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        obj = context.object
        layout = self.layout
        s = _ready(layout, obj)
        if s is None:
            return
        layout.prop(s, "widget_scale")
        col = layout.column(align=True)
        for g in WIDGET_GROUPS:
            row_data = s.group(g.key)
            if row_data is None:
                continue
            row = col.row(align=True)
            row.label(text=g.label, icon=g.icon)
            row.prop(row_data, "scale", text="")
            row.prop(row_data, "offset", text="")

        bone = context.active_pose_bone
        entry = s.control(bone.name) if bone else None
        box = layout.box()
        if entry is None:
            box.label(text="Select a control to size it on its own", icon="INFO")
        else:
            group = group_of(entry.bone)
            box.label(text="%s  (%s)" % (entry.bone, group.label if group else "-"),
                      icon="BONE_DATA")
            box.prop(entry, "scale")
            box.prop(entry, "offset")
            box.operator("efb.reset_widget", icon="LOOP_BACK").bone = entry.bone
        op = layout.operator("efb.reset_widget", text="Reset All Sizes", icon="LOOP_BACK")
        op.everything = True


class EFB_PT_seam(_RigPanel):
    bl_label = "Seam Slide"
    bl_idname = "EFB_PT_seam"
    bl_parent_id = "EFB_PT_rig"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        s = rigedit.settings(context.object)
        if s is not None and s.initialised:
            self.layout.prop(s, "seam", text="")

    def draw(self, context):
        obj = context.object
        layout = self.layout
        s = _ready(layout, obj)
        if s is None:
            return
        col = layout.column()
        col.active = s.seam
        col.prop(s, "seam_amount", slider=True)
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="Elbow_* and Knee_* slide along the fold so the", icon="INFO")
        col.label(text="limb's hairline seam stays shut. Epic Fight's own")
        col.label(text="clips carry the same slide, so this exports.")
        col.label(text="Switching it compensates the marker keys, so the")
        col.label(text="pose holds and an import is never slid twice.")


class EFB_PT_tool(_RigPanel):
    bl_label = "Weapon Preview"
    bl_idname = "EFB_PT_tool"
    bl_parent_id = "EFB_PT_rig"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        obj = context.object
        layout = self.layout
        if _ready(layout, obj) is None:
            return
        tool.draw_tool_row(layout, obj)


class EFB_PT_maintenance(_RigPanel):
    bl_label = "Rig Maintenance"
    bl_idname = "EFB_PT_maintenance"
    bl_parent_id = "EFB_PT_rig"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        obj = context.object
        layout = self.layout
        row = layout.row()
        row.scale_y = 1.4
        row.operator("efb.rig_apply", icon="FILE_REFRESH")
        layout.operator("efb.rig_reset_settings", icon="LOOP_BACK")
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="Apply keeps the pose and the action.", icon="CHECKMARK")
        col.label(text="Generate a new rig for:")
        for line in rigedit.NEEDS_REBUILD:
            col.label(text="  " + line)
        if rootcarry.needs_carrying(obj):
            box = layout.box()
            box.label(text="The root does not carry everything", icon="ERROR")
            box.operator("efb.root_carry", icon="CON_CHILDOF")
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="%d IK handle(s) and %d pole(s) are loose in"
                           % (len(rootcarry.loose_handles(obj)),
                              len(rootcarry.loose_poles(obj))))
            col.label(text="armature space, so moving CTRL-Master")
            col.label(text="leaves them behind. The animation is kept.")
        layout.separator()
        layout.operator("efb.migrate_append_rig", icon="APPEND_BLEND")
        layout.operator("efb.migrate_action", icon="TRACKING_FORWARDS")
        col = layout.column(align=True)
        col.scale_y = 0.8
        found = len(migrate.candidates(context)[0])
        if found:
            col.label(text="%d older rig(s) in this file to migrate from." % found,
                      icon="CHECKMARK")
        else:
            col.label(text="The old rig is usually in its own .blend:", icon="INFO")
            col.label(text="append it first, then migrate from it.")
        col.label(text="It reads the old rig and reports what was lost.")


class EFB_PT_build(bpy.types.Panel):
    bl_label = "Build"
    bl_idname = "EFB_PT_build"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator_menu_enum("efb.generate_rig", "variant", icon="ARMATURE_DATA")
        col.operator("efb.import_animation", text="Import Animation", icon="IMPORT")


CLASSES = (EFB_PT_rig, EFB_PT_arms, EFB_PT_legs, EFB_PT_body, EFB_PT_visibility,
           EFB_PT_limits, EFB_PT_widgets, EFB_PT_seam, EFB_PT_tool, EFB_PT_maintenance,
           EFB_PT_build, EFB_PT_item)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
