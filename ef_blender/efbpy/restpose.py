"""Rest pose override: make the pose the animator has typed in the new zero.

Nothing here touches an edit-bone rest matrix. Each posed control gets a hidden duplicate
that holds the pose, the control itself is zeroed, and a local Copy Transforms at the top
of its stack puts the pose back. So Alt-R and Reset Pose land on the animator's neutral
instead of Epic Fight's, and the twenty deform rests are exactly where biped.json put them.

The duplicates are not Epic Fight joints, so the export walk cannot see them - it takes
its bone list from the armature json, not from the armature.
"""

import math

import bpy
from mathutils import Matrix

from . import snap

PREFIX = "REST-"
OF = "efb_rest_of"
MARK = "efb_rest_override"
CONSTRAINT = "Rest Override"
COLLECTION = "Mechanism"

_IDENTITY = Matrix.Identity(4)


def is_override(pose_bone) -> bool:
    return OF in pose_bone.bone


def overridden(rig):
    """control name -> duplicate name, for every pair currently on the rig."""
    return {pb.name: pb.bone[MARK] for pb in rig.pose.bones
            if MARK in pb.bone and pb.bone[MARK] in rig.pose.bones}


def controls(rig):
    """The bones an override may be minted for. A widget is what makes a bone a control;
    deform bones and the mechanism are neither posed by hand nor safe to duplicate."""
    return [pb for pb in rig.pose.bones
            if pb.custom_shape is not None and not pb.bone.use_deform
            and not is_override(pb)]


def local_offset(pose_bone):
    """The override's contribution in this bone's own local space, or None. The duplicate
    shares the control's rest and parent, so the two local spaces are the same one."""
    mark = pose_bone.bone.get(MARK)
    if not mark:
        return None
    dup = pose_bone.id_data.pose.bones.get(mark)
    return None if dup is None else dup.matrix_basis.copy()


def set_matrix(pose_bone, matrix):
    """Write a world matrix onto a control that may be holding an override.

    pose_bone.matrix ignores the bone's own constraints, and the override is a local Copy
    Transforms at index 0, so a plain write lands displaced by exactly that constraint.
    """
    pose_bone.matrix = matrix
    off = local_offset(pose_bone)
    if off is not None:
        pose_bone.matrix_basis = off.inverted() @ pose_bone.matrix_basis


def _off_rest(pb, tolerance=1e-6) -> bool:
    m = pb.matrix_basis
    return any(abs(m[r][c] - _IDENTITY[r][c]) > tolerance
               for r in range(4) for c in range(4))


def update_rest_pose(rig, context=None, names=None):
    """Fold the current pose of every posed control into its override, or only the named
    ones. Returns the pair map. Running it twice composes rather than replacing, so a
    second adjustment on top of an override lands where the animator sees it."""
    context = context or bpy.context
    pairs = overridden(rig)
    wanted = {pb.name: pb.matrix_basis.copy() for pb in controls(rig)
              if _off_rest(pb) and (names is None or pb.name in names)}
    if not wanted:
        return pairs

    fresh = [n for n in wanted if n not in pairs]
    if fresh:
        _mint(rig, fresh, context)
        pairs = overridden(rig)

    for name, basis in wanted.items():
        dup = rig.pose.bones[pairs[name]]
        dup.matrix_basis = dup.matrix_basis @ basis
        rig.pose.bones[name].matrix_basis = _IDENTITY.copy()
    rig.update_tag()
    context.view_layer.update()
    return pairs


def _mint(rig, names, context):
    """Duplicate bones at the controls' own rests, then wire each one up."""
    was = rig.mode
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        made = []
        for name in names:
            src = rig.data.edit_bones[name]
            eb = rig.data.edit_bones.new(PREFIX + name)
            eb.head, eb.tail, eb.roll = src.head, src.tail, src.roll
            eb.parent = src.parent
            eb.use_deform = False
            eb.use_connect = False
            made.append((name, eb.name))
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    coll = rig.data.collections_all.get(COLLECTION)
    for name, dup_name in made:
        bone, dup = rig.data.bones[name], rig.data.bones[dup_name]
        if coll is not None:
            coll.assign(dup)
        dup.hide = True
        dup.hide_select = True
        dup[OF] = name
        bone[MARK] = dup_name
        pb = rig.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        rig.pose.bones[dup_name].rotation_mode = "QUATERNION"
        con = pb.constraints.get(CONSTRAINT) or pb.constraints.new("COPY_TRANSFORMS")
        con.name = CONSTRAINT
        con.target = rig
        con.subtarget = dup_name
        con.target_space = con.owner_space = "LOCAL"
        con.mix_mode = "BEFORE_FULL"
        pb.constraints.move(len(pb.constraints) - 1, 0)
    if was == "POSE":
        bpy.ops.object.mode_set(mode="POSE")
    context.view_layer.update()


def reset_rest_pose(rig, context=None):
    """Take the overrides out and keep the pose they were producing. Measured back onto
    each control rather than recomputed, so nothing is lost to the removal.

    Returns (pairs removed, how much pose would not go back on).
    """
    context = context or bpy.context
    pairs = overridden(rig)
    if not pairs:
        return 0, 0.0
    context.view_layer.update()
    held = {n: rig.pose.bones[n].matrix.copy() for n in pairs}

    for name, dup_name in pairs.items():
        pb = rig.pose.bones[name]
        con = pb.constraints.get(CONSTRAINT)
        if con is not None:
            pb.constraints.remove(con)
        del pb.bone[MARK]

    was = rig.mode
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for dup_name in pairs.values():
            eb = rig.data.edit_bones.get(dup_name)
            if eb is not None:
                rig.data.edit_bones.remove(eb)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    if was == "POSE":
        bpy.ops.object.mode_set(mode="POSE")

    context.view_layer.update()
    return len(pairs), _restore(rig, held, context)


_PASSES = 8
_SETTLED = 1e-6


def _restore(rig, held, context):
    """Write every control back onto the pose it was producing, and return what is left
    over.

    Not in bone-parent order. A control can hang off a DEFORM bone - CTRL-Torso's parent
    is the Root joint, which a constraint drives from CTRL-Root - so parent depth is not
    the order the rig resolves in, and sorting by it writes some children before the
    ancestors they really follow. Repeated passes converge without modelling the graph.
    """
    worst = 0.0
    for _ in range(_PASSES):
        for name, matrix in held.items():
            rig.pose.bones[name].matrix = matrix
            context.view_layer.update()
        worst = max((_gap(rig.pose.bones[n].matrix, m) for n, m in held.items()),
                    default=0.0)
        if worst <= _SETTLED:
            break
    return worst


def _gap(a, b):
    return max(abs(a[r][c] - b[r][c]) for r in range(4) for c in range(4))



STRAIGHT_ENOUGH = 0.5


def limb_bend(rig, limb) -> float:
    """Degrees between the upper joint's own axis and the lower joint's, as posed."""
    up, low = rig.pose.bones.get(limb.upper), rig.pose.bones.get(limb.lower)
    if up is None or low is None:
        return 0.0
    a = up.matrix.to_3x3().col[1].normalized()
    b = low.matrix.to_3x3().col[1].normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, a.dot(b)))))


def straighten_arms(rig, context=None) -> dict:
    """Roll each bent forearm onto its upper joint's axis and make that the new zero.

    Epic Fight's 6.312 degrees are in the rest matrices and cannot leave them. What moves
    is where clearing a control lands, which is the override's whole job. The angle is
    measured live rather than typed, so a rig built at other proportions gets its own
    number. Returns {limb key: degrees taken out}.

    Measured at the cleared control and not at the pose on screen. Reading the live bend
    instead made the button mean something different on every frame - on an imported clip
    it took 45.9 degrees out of a leg, because at that frame the leg really was bent that
    far. It is the zero this moves, so the zero is what it has to measure.
    """
    from .ops import rig_limbs

    context = context or bpy.context
    out, held = {}, {}
    for limb in rig_limbs(rig):
        pb = rig.pose.bones.get(limb.fk_lower)
        if pb is None:
            continue
        pose = pb.matrix_basis.copy()
        pb.matrix_basis = _IDENTITY.copy()
        rig.update_tag()
        context.view_layer.update()
        before = limb_bend(rig, limb)
        if before <= STRAIGHT_ENOUGH:
            pb.matrix_basis = pose
            continue
        pb.rotation_mode = "QUATERNION"
        best, kept = before, _IDENTITY.copy()
        for sign in (1.0, -1.0):
            pb.matrix_basis = Matrix.Rotation(math.radians(sign * before), 4, "X")
            rig.update_tag()
            context.view_layer.update()
            got = limb_bend(rig, limb)
            if got < best:
                best, kept = got, pb.matrix_basis.copy()
        pb.matrix_basis = kept
        held[limb.fk_lower] = pose
        out[limb.key] = before - best
    rig.update_tag()
    context.view_layer.update()
    if not out:
        return out
    update_rest_pose(rig, context, names=list(held))
    for name, pose in held.items():
        rig.pose.bones[name].matrix_basis = pose
    rig.update_tag()
    context.view_layer.update()
    return out


def keyed_controls(rig, names):
    """Which of `names` an action already keys - moving their zero shifts every key."""
    from .animscene import action_curves

    action = rig.animation_data.action if rig.animation_data else None
    if action is None:
        return []
    paths = {'pose.bones["%s"].' % bpy.utils.escape_identifier(n): n for n in names}
    return sorted({n for fc in action_curves(action, rig)
                   for p, n in paths.items() if fc.data_path.startswith(p)})


class _Base(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)


class EFB_OT_update_rest_pose(_Base):
    """Make the current pose the rig's zero. Clearing a control returns here instead of to
    the Epic Fight rest; the twenty joint rest matrices are not touched"""
    bl_idname = "efb.update_rest_pose"
    bl_label = "Update Rest Pose"

    def execute(self, context):
        rig = context.object
        pairs = update_rest_pose(rig, context)
        if not pairs:
            self.report({"INFO"}, "nothing is posed - the rest is already the pose")
        else:
            self.report({"INFO"}, "%d controls hold the new rest" % len(pairs))
        return {"FINISHED"}


class EFB_OT_reset_rest_pose(_Base):
    """Put the zero back to Epic Fight's rest, keeping the pose that is on screen"""
    bl_idname = "efb.reset_rest_pose"
    bl_label = "Reset Rest Pose"

    def execute(self, context):
        n, left = reset_rest_pose(context.object, context)
        if left > _SETTLED:
            self.report({"WARNING"},
                        "%d overrides removed, %.3g m of pose a joint stop would not "
                        "take back" % (n, left))
        else:
            self.report({"INFO"}, "%d overrides removed, pose kept" % n)
        return {"FINISHED"}


class EFB_OT_straighten_arms(_Base):
    """Take Epic Fight's 6.3 degree forearm bend out of the animator's zero, so clearing a
    control lands on a straight arm. The arm on screen straightens with it. The 20 joint
    rest matrices are not touched, and Reset puts the zero back"""
    bl_idname = "efb.straighten_arms"
    bl_label = "Straighten Arms"

    def execute(self, context):
        from .ops import rig_limbs

        rig = context.object
        lower = [l.fk_lower for l in rig_limbs(rig) if l.fk_lower]
        took = straighten_arms(rig, context)
        if not took:
            self.report({"INFO"}, "both arms are already straight")
            return {"FINISHED"}
        self.report({"INFO"}, "straightened %s"
                    % ", ".join("%s by %.2f deg" % (k, v)
                                for k, v in sorted(took.items())))
        keyed = keyed_controls(rig, lower)
        if keyed:
            self.report({"WARNING"},
                        "%s already carry keys - the clip now plays against the new zero. "
                        "Reset Rest Pose puts it back" % ", ".join(keyed))
        return {"FINISHED"}


CLASSES = (EFB_OT_update_rest_pose, EFB_OT_reset_rest_pose, EFB_OT_straighten_arms)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
