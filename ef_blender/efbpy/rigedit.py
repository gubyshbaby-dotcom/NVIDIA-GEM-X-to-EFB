"""Editable rig settings and the code that applies them.

The settings live in a PropertyGroup on the rig OBJECT, so they are saved with the file
and two rigs in one .blend keep their own. Nothing here is a preference.

Everything an apply touches is pose-level: custom shapes, constraints and transform
locks. None of it is an edit bone and none of it is an fcurve, so re-applying is safe with
a half-finished animation loaded. What genuinely needs the rig regenerating is listed in
`NEEDS_REBUILD`.
"""

import math

import bpy
from bpy.props import (BoolProperty, BoolVectorProperty, CollectionProperty,
                       EnumProperty, FloatProperty, FloatVectorProperty,
                       PointerProperty, StringProperty)

from efb.rigdef import HINGE_LOCK, LOCK_LIMB_ROTATION
from efb.rigedit import (AXIS_GROUPS, AXIS_LIMITS, DEFORM_PIN, DEFORM_ROLL_PIN,
                         EF_PRESET, HINGE_CONSTRAINT, HUMAN_PRESET, LIMIT_CONSTRAINT,
                         PIN_AXES, PIN_CONSTRAINT, REPLAY_PROP, ROLL_PIN_ORDER,
                         SEAM_CONSTRAINT, TOOL_COLLECTION, TOOL_SOCKETS, WIDGET_GROUPS,
                         axis_ranges, group_of, hinges, seams)

from . import snap
from .animscene import (imported_action, location_keys, offset_location_keys,
                        seam_controls, seam_shift)
from .ops import rig_limbs

NEEDS_REBUILD = (
    "switching the skeleton between wide and slim arms",
    "moving a control's pivot - keys are stored against the rest it had when they were made",
    "adding or removing controls",
    "a new add-on version's elbow fold reference - a rig keeps the one it was built with",
)

_APPLYING = [False]


def settings(rig):
    """The settings block, or None when the addon is not registered."""
    return getattr(rig, "efb_rig", None)


def pinned(rig, key) -> bool:
    """Is this limb's lower segment held to its fold axis right now?

    Four switches can let the roll through and the panel, the gates and the drivers all
    have to agree on which: the master stop, the limb's own stop, its pin, and Replay.
    """
    s = settings(rig)
    row = s.limb(key) if s is not None else None
    return bool(row is not None and s.limits and row.limits and row.pin and not s.replay)




class EFB_PG_control(bpy.types.PropertyGroup):
    """One widget-bearing control: what the generator drew, plus the animator's tweak."""
    bone: StringProperty()
    base_scale: FloatVectorProperty(size=3, default=(1.0, 1.0, 1.0))
    base_translation: FloatVectorProperty(size=3)
    base_rotation: FloatVectorProperty(size=3)
    scale: FloatProperty(name="Size", default=1.0, min=0.05, max=8.0,
                         description="Multiplies this one control's widget",
                         update=lambda s, c: apply_widgets(s.id_data))
    offset: FloatVectorProperty(name="Offset", size=3, subtype="TRANSLATION",
                                description="Nudge, in the bone's own space",
                                update=lambda s, c: apply_widgets(s.id_data))


class EFB_PG_group(bpy.types.PropertyGroup):
    key: StringProperty()
    scale: FloatProperty(name="Size", default=1.0, min=0.05, max=8.0,
                         update=lambda s, c: apply_widgets(s.id_data))
    offset: FloatProperty(name="Along bone", default=0.0, min=-1.0, max=1.0,
                          description="Slide the whole group along its bones",
                          update=lambda s, c: apply_widgets(s.id_data))


_BAND_HELP = ("How far this joint may fold, in degrees of the Epic Fight joint's own "
              "rest. The FK control, the IK solver's chain and the exported joint all "
              "stop here - there is no second number anywhere. Narrowing it takes reach "
              "away from the IK handle, which is the stop doing its job")


class EFB_PG_limb(bpy.types.PropertyGroup):
    key: StringProperty()
    label: StringProperty()
    limits: BoolProperty(name="Limit", default=True,
                         description="Stop this joint folding the wrong way",
                         update=lambda s, c: apply_limits(s.id_data))
    hinge_min: FloatProperty(name="Min", default=-math.pi * 0.78,
                             min=-math.pi, max=math.pi, subtype="ANGLE",
                             description=_BAND_HELP,
                             update=lambda s, c: apply_limits(s.id_data))
    hinge_max: FloatProperty(name="Max", default=0.0, min=-math.pi, max=math.pi,
                             subtype="ANGLE", description=_BAND_HELP,
                             update=lambda s, c: apply_limits(s.id_data))
    pin: BoolProperty(
        name="Fold on one axis only", default=True,
        description="Hold this joint's twist and sideways fold at zero, so it can only "
                    "fold the way an elbow or a knee folds. The seam between the two "
                    "boxes of the limb only closes on the fold axis, so a sideways fold "
                    "tears it open and the boxes cut into each other. The reference rig "
                    "holds the same two channels at zero on every path",
        update=lambda s, c: apply_limits(s.id_data))


class EFB_PG_axis(bpy.types.PropertyGroup):
    """One non-hinge control's rotation box. `used` is per axis, so a joint can be held
    on two axes and left free on the third."""
    bone: StringProperty()
    group: StringProperty()
    note: StringProperty()
    used: BoolVectorProperty(size=3, default=(False, False, False),
                             update=lambda s, c: apply_axis_limits(s.id_data))
    minimum: FloatVectorProperty(size=3, subtype="EULER",
                                 update=lambda s, c: apply_axis_limits(s.id_data))
    maximum: FloatVectorProperty(size=3, subtype="EULER",
                                 update=lambda s, c: apply_axis_limits(s.id_data))


class EFB_PG_rig(bpy.types.PropertyGroup):
    initialised: BoolProperty(default=False)

    advanced: BoolProperty(
        name="Advanced", default=False,
        description="Show the rows an animator needs occasionally: the pole space, the "
                    "end locks, rest pose and mirror")

    controls: CollectionProperty(type=EFB_PG_control)
    groups: CollectionProperty(type=EFB_PG_group)
    limbs: CollectionProperty(type=EFB_PG_limb)
    axes: CollectionProperty(type=EFB_PG_axis)

    widget_scale: FloatProperty(
        name="All widgets", default=1.0, min=0.1, max=5.0,
        description="Scales every control at once",
        update=lambda s, c: apply_widgets(s.id_data))

    limits: BoolProperty(
        name="Joint Limits", default=True,
        description="Master switch for every joint stop on this rig",
        update=lambda s, c: apply_every_limit(s.id_data))
    replay: BoolProperty(
        name="Replay this clip as it was written", default=False,
        description="Stand the off-axis pin down so an imported clip plays back and "
                    "exports exactly as it arrived. Import raises this by itself when "
                    "the clip it read rolls a forearm or a shin, because clamping "
                    "somebody's file on the way in would be a lie. Turn it off to go "
                    "back to authoring against the pin",
        update=lambda s, c: apply_limits(s.id_data))
    preset: EnumProperty(
        name="Ranges", default=EF_PRESET,
        items=[(EF_PRESET, "Epic Fight", "Everything the shipped clips do, and no more. "
                "Nothing upstream animates is ever clamped"),
               (HUMAN_PRESET, "Human", "Anatomical ranges. Tighter than several shipped "
                "clips, which will be clamped on import")],
        update=lambda s, c: load_preset(s.id_data))
    axis_spine: BoolProperty(
        name="Spine and neck", default=AXIS_GROUPS["spine"][2],
        update=lambda s, c: apply_axis_limits(s.id_data))
    axis_clavicle: BoolProperty(
        name="Collarbones", default=AXIS_GROUPS["clavicle"][2],
        update=lambda s, c: apply_axis_limits(s.id_data))
    axis_ball: BoolProperty(
        name="Shoulders and hips", default=AXIS_GROUPS["ball"][2],
        description="The upper arm is never limited - it passes through gimbal in every "
                    "euler order the shipped clips are measured in",
        update=lambda s, c: apply_axis_limits(s.id_data))

    seam: BoolProperty(
        name="Seam Slide", default=True,
        description="Slide Elbow_* and Knee_* along the fold so the limb's hairline seam "
                    "stays shut. This is how Epic Fight's own clips bend, and it exports. "
                    "Switching it moves the marker keys the other way, so the pose you "
                    "can see never changes and an imported clip is never slid twice",
        update=lambda s, c: apply_seam(s.id_data))
    seam_amount: FloatProperty(
        name="Amount", default=1.0, min=0.0, max=1.5,
        description="Scales the measured slide. 1.0 is the reference rig's",
        update=lambda s, c: apply_seam(s.id_data))

    hide_tools: BoolProperty(
        name="Hide tool controls", default=False,
        description="Take the weapon sockets out of the viewport for this session",
        update=lambda s, c: apply_visibility(s.id_data))

    show_tool_proxy: BoolProperty(
        name="Show tool direction", default=False,
        description="Put a stick down each weapon socket, pointing the way the tool "
                    "points. Viewport only - it does not render and is not exported",
        update=lambda s, c: _apply_tool_proxy(s.id_data, c))

    tool_object: PointerProperty(
        type=bpy.types.Object, name="Weapon",
        description="An object to hang on a socket, to see how a weapon will sit",
        poll=lambda s, o: o.type in {"MESH", "EMPTY", "CURVE"})
    tool_socket: EnumProperty(
        name="Socket", items=[(n, n, "") for n in TOOL_SOCKETS], default=TOOL_SOCKETS[0])

    def limb(self, key):
        return self.limbs.get(key)

    def axis(self, bone):
        return self.axes.get(bone)

    def group_on(self, key):
        return getattr(self, "axis_" + key, False)

    def group(self, key):
        return self.groups.get(key)

    def control(self, bone):
        return self.controls.get(bone)




def ensure_rows(rig):
    """Fills the collections from whatever the rig actually has. Idempotent; the baseline
    is captured once, on the first pass, and is what Reset goes back to."""
    s = settings(rig)
    if s is None:
        return None
    for g in WIDGET_GROUPS:
        if s.group(g.key) is None:
            row = s.groups.add()
            row.name = row.key = g.key
    for pb in rig.pose.bones:
        if pb.custom_shape is None or s.control(pb.name) is not None:
            continue
        row = s.controls.add()
        row.name = row.bone = pb.name
        row.base_scale = tuple(pb.custom_shape_scale_xyz)
        row.base_translation = tuple(pb.custom_shape_translation)
        row.base_rotation = tuple(pb.custom_shape_rotation_euler)
    for h in hinges(rig_limbs(rig)):
        if s.limb(h.limb) is not None:
            continue
        row = s.limbs.add()
        row.name = row.key = h.limb
        row.label = h.limb.replace("_", ".").upper()
        row.limits = h.enabled
        row.hinge_min = math.radians(h.minimum)
        row.hinge_max = math.radians(h.maximum)
    for spec in AXIS_LIMITS:
        if s.axis(spec.bone) is not None or spec.bone not in rig.pose.bones:
            continue
        row = s.axes.add()
        row.name = row.bone = spec.bone
        row.group = spec.group
        row.note = spec.note
        _load_axis_row(row, spec, s.preset)
    s.initialised = True
    return s


def _load_axis_row(row, spec, preset):
    ranges = axis_ranges(spec, preset)
    row.used = [r is not None for r in ranges]
    row.minimum = [math.radians(r[0]) if r else 0.0 for r in ranges]
    row.maximum = [math.radians(r[1]) if r else 0.0 for r in ranges]


def load_preset(rig):
    """Rewrites every editable range from the chosen preset. The switches themselves -
    which groups are on, which limbs are hinged - are left alone."""
    s = settings(rig)
    if s is None or _APPLYING[0]:
        return
    _APPLYING[0] = True
    try:
        by_bone = {a.bone: a for a in AXIS_LIMITS}
        for row in s.axes:
            spec = by_bone.get(row.bone)
            if spec is not None:
                _load_axis_row(row, spec, s.preset)
        for h in hinges(rig_limbs(rig)):
            row, band = s.limb(h.limb), _hinge_band(h, s.preset)
            if row is not None:
                row.hinge_min, row.hinge_max = math.radians(band[0]), math.radians(band[1])
    finally:
        _APPLYING[0] = False
    apply_limits(rig)
    apply_axis_limits(rig)


HUMAN_HINGE = {"arm": (-10.0, 145.0), "leg": (-150.0, 0.0)}


def _hinge_band(h, preset):
    if preset == HUMAN_PRESET:
        return HUMAN_HINGE["arm" if h.limb.startswith("arm") else "leg"]
    return (h.minimum, h.maximum)




def apply_widgets(rig):
    s = settings(rig)
    if s is None or _APPLYING[0]:
        return
    for row in s.controls:
        pb = rig.pose.bones.get(row.bone)
        if pb is None or pb.custom_shape is None:
            continue
        group = group_of(row.bone)
        g = s.group(group.key) if group else None
        factor = s.widget_scale * row.scale * (g.scale if g else 1.0)
        pb.custom_shape_scale_xyz = [v * factor for v in row.base_scale]
        along = g.offset if g else 0.0
        pb.custom_shape_translation = (
            row.base_translation[0] + row.offset[0],
            row.base_translation[1] + row.offset[1] + along,
            row.base_translation[2] + row.offset[2])


def _prop_driver(rig, path, expression, prop, index=-1, also=()):
    """One scripted driver on the rig object reading its own properties. Replaces
    whatever was on that path already. `prop` is the variable `v`; `also` names further
    (variable, property) pairs the expression may use."""
    fcurve = rig.driver_add(path, index) if index >= 0 else rig.driver_add(path)
    for mod in list(fcurve.modifiers):
        fcurve.modifiers.remove(mod)
    drv = fcurve.driver
    drv.type = "SCRIPTED"
    for existing in list(drv.variables):
        drv.variables.remove(existing)
    for name, source in (("v", prop),) + tuple(also):
        var = drv.variables.new()
        var.name = name
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "OBJECT"
        var.targets[0].id = rig
        var.targets[0].data_path = '["%s"]' % source
    drv.expression = expression
    return fcurve


def _drop_driver(rig, path, index=-1):
    if rig.animation_data is None:
        return
    try:
        rig.driver_remove(path, index)
    except (TypeError, RuntimeError):
        pass


def _influence_path(bone, constraint):
    return 'pose.bones["%s"].constraints["%s"].influence' % (bone, constraint)


def _gate_by_lock(rig, bone, constraint, on):
    """The master switch owns the stop's influence, not the panel. A driven object
    property keys, and it still holds with the addon disabled."""
    path = _influence_path(bone, constraint)
    _drop_driver(rig, path)
    if on and LOCK_LIMB_ROTATION in rig:
        _prop_driver(rig, path, "v", LOCK_LIMB_ROTATION)


def _gate_the_pin(rig, bone, constraint, on):
    """The pin answers to two switches, not one: the master stop turns it off like every
    other limit, and Replay stands it down so an imported clip exports as it arrived."""
    path = _influence_path(bone, constraint)
    _drop_driver(rig, path)
    if on and LOCK_LIMB_ROTATION in rig:
        _prop_driver(rig, path, "v * (1.0 - r)", LOCK_LIMB_ROTATION,
                     also=(("r", REPLAY_PROP),))


def _restore_the_roll(rig, limb):
    """Put the wrist / ankle lock's driver back to the plain `ik * lock` the generator
    writes. 0.16.0 gated this on the pin instead of clamping downstream, which let the
    upper segment's roll leak into the lower one through the parent - 12.37 degrees on an
    arm. DEFORM_ROLL_PIN takes the roll off after the solve now, so this driver has no
    business knowing about the pin.
    """
    pb = rig.pose.bones.get(limb.lower)
    con = pb.constraints.get(ROLL_CONSTRAINT) if pb is not None else None
    if con is None:
        return
    path = _influence_path(limb.lower, ROLL_CONSTRAINT)
    _drop_driver(rig, path)
    _prop_driver(rig, path, "v * lock", limb.prop, also=(("lock", limb.end_lock),))


def apply_limits(rig):
    s = settings(rig)
    if s is None or _APPLYING[0]:
        return
    rig[REPLAY_PROP] = 1.0 if s.replay else 0.0
    limbs = {l.key: l for l in rig_limbs(rig)}
    for h in hinges(rig_limbs(rig)):
        row = s.limb(h.limb)
        pb = rig.pose.bones.get(h.bone)
        if pb is None or row is None:
            continue
        for name in (HINGE_CONSTRAINT, PIN_CONSTRAINT):
            old = pb.constraints.get(name)
            if old is not None:
                pb.constraints.remove(old)
        on = s.limits and row.limits
        held = list(PIN_AXES)
        for i in range(3):
            _drop_driver(rig, 'pose.bones["%s"].lock_rotation' % h.bone, i)
        lock = [False, False, False]
        if on:
            con = pb.constraints.new("LIMIT_ROTATION")
            con.name = HINGE_CONSTRAINT
            con.owner_space = "LOCAL"
            con.use_transform_limit = True
            con.euler_order = "XYZ"
            con.use_limit_x = True
            con.min_x, con.max_x = row.hinge_min, row.hinge_max
            for i, band in zip((1, 2), h.slack or (None, None)):
                if band is None or row.pin or i not in held:
                    continue
                _band(con, i, math.radians(band[0]), math.radians(band[1]))
            if row.pin:
                _pin_constraint(pb, held)
                for i in held:
                    lock[i] = True
        pb.lock_rotation = lock
        _gate_by_lock(rig, h.bone, HINGE_CONSTRAINT, on)
        _gate_the_pin(rig, h.bone, PIN_CONSTRAINT, on and row.pin)
        if h.limb in limbs:
            _restore_the_roll(rig, limbs[h.limb])
        if on and row.pin and LOCK_LIMB_ROTATION in rig:
            for i in held:
                _prop_driver(rig, 'pose.bones["%s"].lock_rotation' % h.bone,
                             "v > 0.5", LOCK_LIMB_ROTATION, index=i)
        _apply_deform_limit(rig, h, row, on)
        _apply_ik_limit(rig, h, row, on)


def _pin_constraint(pb, held):
    """Local Y and Z held at zero, so the joint can only fold the way it folds.

    Its own constraint and not three more axes on the hinge: the fold stop applies
    always, the pin steps aside for Replay, and one constraint carries one influence.
    """
    con = pb.constraints.new("LIMIT_ROTATION")
    con.name = PIN_CONSTRAINT
    con.owner_space = "LOCAL"
    con.use_transform_limit = True
    con.euler_order = "XYZ"
    con.use_limit_z = True
    con.min_z = con.max_z = 0.0
    con.use_limit_y = 1 in held
    con.min_y = con.max_y = 0.0
    return con


def _band(con, axis, lo, hi):
    """One axis of a Limit Rotation, by index."""
    letter = "xyz"[axis]
    setattr(con, "use_limit_" + letter, True)
    setattr(con, "min_" + letter, lo)
    setattr(con, "max_" + letter, hi)


def apply_axis_limits(rig):
    """The non-hinge boxes: spine, neck, collarbones, hips. One constraint per control,
    removed outright when the group is off, so nothing dead is left in the stack."""
    s = settings(rig)
    if s is None or _APPLYING[0]:
        return
    for row in s.axes:
        pb = rig.pose.bones.get(row.bone)
        if pb is None:
            continue
        old = pb.constraints.get(LIMIT_CONSTRAINT)
        if old is not None:
            pb.constraints.remove(old)
        spec = next((a for a in AXIS_LIMITS if a.bone == row.bone), None)
        if not (s.limits and s.group_on(row.group)) or spec is None:
            continue
        if not any(row.used):
            continue
        con = pb.constraints.new("LIMIT_ROTATION")
        con.name = LIMIT_CONSTRAINT
        con.owner_space = "LOCAL"
        con.use_transform_limit = True
        con.euler_order = spec.order
        for i in range(3):
            if row.used[i]:
                _band(con, i, row.minimum[i], row.maximum[i])


def apply_every_limit(rig):
    apply_limits(rig)
    apply_axis_limits(rig)


def apply_seam(rig):
    """Elbow_* and Knee_* slide along the fold below them, which is what keeps the two
    boxes of a limb from tearing apart at the hairline. It changes the exported file, and
    it has to: Epic Fight's own clips carry the same slide.

    Pose-preserving on an imported clip: those carry Epic Fight's own slide in their
    marker keys and had the constraint's copy taken back out at import, so flipping the
    toggle afterwards has to move the keys the other way or the elbow slides twice. On a
    hand-animated rig there is nothing to compensate and the toggle does what it says.
    """
    s = settings(rig)
    if s is None or _APPLYING[0]:
        return
    action = imported_action(rig) if _seam_changing(rig, s) else None
    pairs = seam_controls(rig) if action is not None else {}
    frames = {f for c in pairs.values() for f in location_keys(action, rig, c)}
    before = seam_shift(rig, pairs, frames) if frames else {}

    for seam in seams(rig.pose.bones.keys()):
        pb = rig.pose.bones[seam.marker]
        old = pb.constraints.get(SEAM_CONSTRAINT)
        if old is not None:
            pb.constraints.remove(old)
        if _seam_target(seam, s) is None:
            continue
        con = pb.constraints.new("TRANSFORM")
        con.name = SEAM_CONSTRAINT
        con.target = rig
        con.subtarget = seam.lower
        con.map_from, con.map_to = "ROTATION", "LOCATION"
        con.target_space = con.owner_space = "LOCAL"
        con.mix_mode = "ADD"
        con.from_min_x_rot = math.radians(seam.from_x[0])
        con.from_max_x_rot = math.radians(seam.from_x[1])
        con.map_to_x_from = con.map_to_y_from = con.map_to_z_from = "X"
        con.to_min_z, con.to_max_z = _seam_target(seam, s)

    if frames:
        _hold_the_seam_pose(rig, pairs, frames, before, action)


def _seam_target(seam, s):
    """The two mapped ends this seam wants, or None when the slide is off."""
    if not s.seam or s.seam_amount <= 0.0:
        return None
    return (seam.to_z[0] * s.seam_amount, seam.to_z[1] * s.seam_amount)


def _seam_changing(rig, s) -> bool:
    """Whether this apply will really move a seam. Every other apply goes through here
    too, and the pose-preserving pass steps the scene through every key, so an apply that
    changes nothing must not pay for it."""
    for seam in seams(rig.pose.bones.keys()):
        want = _seam_target(seam, s)
        con = rig.pose.bones[seam.marker].constraints.get(SEAM_CONSTRAINT)
        if (want is None) != (con is None):
            return True
        if want is not None and (abs(want[0] - con.to_min_z) > 1e-6
                                 or abs(want[1] - con.to_max_z) > 1e-6):
            return True
    return False


def _hold_the_seam_pose(rig, pairs, frames, before, action):
    """Give each marker's control back exactly what the constraint stopped adding.

    Measured on both sides rather than predicted, so it holds for a change of Amount as
    well as an on/off.
    """
    rig.update_tag()
    bpy.context.view_layer.update()
    after = seam_shift(rig, pairs, frames)
    offset_location_keys(rig, action, pairs,
                         {m: {f: before[m][f] - v for f, v in per.items()}
                          for m, per in after.items()})


DEFORM_LIMIT = "Hinge (export)"

DEFORM_LIMIT_SOLVED = "Hinge (export, after IK)"

FK_CONSTRAINT = "FK"

ROLL_CONSTRAINT = "Roll"


def _seat_after_fk(pb):
    """Move the constraint just added to sit right behind the FK copy.

    Left at the end of the stack it clamps a euler the IK handover has already rolled:
    the deform forearm takes its direction from a damped track and its roll from the IK
    handle, so its local X is not the solver's fold and reads -63 to -92 degrees on an
    overhead reach. Clamping that at -35 pulled the hand up to 0.26 off its own handle
    while the solver chain sat on the target exactly.
    """
    names = [c.name for c in pb.constraints]
    handovers = [i for i, n in enumerate(names) if n == FK_CONSTRAINT]
    pb.constraints.move(len(names) - 1, (max(handovers) + 1) if handovers else 0)


def _fold_stop(pb, name, row):
    con = pb.constraints.new("LIMIT_ROTATION")
    con.name = name
    con.owner_space = "LOCAL"
    con.use_transform_limit = True
    con.euler_order = "XYZ"
    con.use_limit_x = True
    con.min_x, con.max_x = row.hinge_min, row.hinge_max
    return con


def _apply_deform_limit(rig, h, row, on):
    """The same stops on the Epic Fight joint itself, so the export is clamped whichever
    control the animator reached for.

    Four constraints, two places. The fold stop and the off-axis pin sit behind the FK
    copy, where under FK they are the final word; under IK the solver's damped track and
    locked track sit below them and are, so DEFORM_ROLL_PIN goes near the end of the stack
    to take the roll off after the solve, and DEFORM_LIMIT_SOLVED repeats the fold band
    right behind it. See _roll_pin_constraint for the euler order that one needs, and
    DEFORM_LIMIT_SOLVED for why the fold band can only be repeated behind it.

    Without the pin there is no roll pin, so nothing de-rolls the euler at the end of the
    stack and the repeat is left off: under IK that limb's fold is held by the solver's own
    band alone.
    """
    pb = rig.pose.bones.get(h.deform)
    if pb is None:
        return
    for name in (DEFORM_LIMIT, DEFORM_LIMIT_SOLVED, DEFORM_PIN, DEFORM_ROLL_PIN):
        old = pb.constraints.get(name)
        if old is not None:
            pb.constraints.remove(old)
    if not on:
        for name in (DEFORM_LIMIT, DEFORM_LIMIT_SOLVED):
            _gate_by_lock(rig, h.deform, name, False)
        _gate_the_pin(rig, h.deform, DEFORM_PIN, False)
        _gate_the_pin(rig, h.deform, DEFORM_ROLL_PIN, False)
        return
    _fold_stop(pb, DEFORM_LIMIT, row)
    _seat_after_fk(pb)
    _gate_by_lock(rig, h.deform, DEFORM_LIMIT, True)
    if row.pin:
        pin = _pin_constraint(pb, PIN_AXES)
        pin.name = DEFORM_PIN
        _seat_after_fk(pb)
        _roll_pin_constraint(pb)
        _fold_stop(pb, DEFORM_LIMIT_SOLVED, row)
    _gate_the_pin(rig, h.deform, DEFORM_LIMIT_SOLVED, row.pin)
    _gate_the_pin(rig, h.deform, DEFORM_PIN, row.pin)
    _gate_the_pin(rig, h.deform, DEFORM_ROLL_PIN, row.pin)


def _roll_pin_constraint(pb):
    """The segment's own-axis roll, held at zero, at the end of the stack.

    A local rotation is aim-then-roll: R = Rz Rx Ry with Y applied first, so in YXZ the Y
    angle IS the roll about the bone's own axis and X and Z are the direction. Clamping Y
    alone there takes the roll off and leaves the solver's aim untouched, which is why
    this cannot be folded into DEFORM_PIN: that one runs in XYZ, where zeroing Y and Z
    after a damped track would move the tip as well.
    """
    con = pb.constraints.new("LIMIT_ROTATION")
    con.name = DEFORM_ROLL_PIN
    con.owner_space = "LOCAL"
    con.use_transform_limit = True
    con.euler_order = ROLL_PIN_ORDER
    con.use_limit_y = True
    con.min_y = con.max_y = 0.0
    return con


def _apply_ik_limit(rig, h, row, on):
    """The panel's band, on the solver's chain. This is the ONLY stop the solve itself
    obeys, so if it does not carry the panel's numbers the panel is printing a number
    nobody enforces - which is what shipped.

    The solver's chain is a one-axis hinge whichever way the panel is set: that is
    rigdef.HINGE_LOCK, and clearing it is what let the elbow invert. Off, the band opens
    to the whole half turn rather than to rigdef's default, or the master switch would
    still leave the solver stopped where a stale panel once said.

    The panel's numbers are against the DEFORM rest and the solver reads its own, which
    the seed kinks by IK_SEED - so the seed comes off both ends here exactly as it does in
    rigdef, or the first touch of this panel silently shifts the reachable band by it.
    """
    ikb = rig.pose.bones.get(h.ik_bone)
    if ikb is None:
        return
    seed = math.radians(h.seed)
    lo, hi = (row.hinge_min - seed, row.hinge_max - seed) if on else (-math.pi, math.pi)
    ikb.lock_ik_x, ikb.lock_ik_y, ikb.lock_ik_z = HINGE_LOCK
    ikb.use_ik_limit_x = on
    ikb.ik_min_x, ikb.ik_max_x = lo, hi


def skinned_meshes(rig):
    return [o for o in bpy.data.objects if o.type == "MESH"
            and any(m.type == "ARMATURE" and m.object is rig for m in o.modifiers)]


def apply_visibility(rig):
    s = settings(rig)
    if s is None or _APPLYING[0]:
        return
    coll = rig.data.collections_all.get(TOOL_COLLECTION)
    if coll is not None:
        coll.is_visible = not s.hide_tools


def _apply_tool_proxy(rig, context=None):
    from . import toolproxy
    if not _APPLYING[0]:
        toolproxy.apply_proxy(rig, context)


def apply_all(rig):
    """The rebuild in place. Pose, action and rest all survive it."""
    ensure_rows(rig)
    apply_widgets(rig)
    apply_limits(rig)
    apply_axis_limits(rig)
    apply_seam(rig)
    apply_visibility(rig)


_RESET = ("widget_scale", "limits", "replay", "preset", "axis_spine",
          "axis_clavicle", "axis_ball", "seam", "seam_amount", "hide_tools", "advanced")


def reset_settings(rig):
    """Back to what a fresh rig has. The widgets go back to their baseline BEFORE the
    rows are dropped, or the next capture would take the scaled sizes as the baseline."""
    s = settings(rig)
    if s is None:
        return
    _APPLYING[0] = True
    try:
        for row in s.controls:
            row.scale, row.offset = 1.0, (0.0, 0.0, 0.0)
        for g in s.groups:
            g.scale, g.offset = 1.0, 0.0
        s.property_unset("widget_scale")
    finally:
        _APPLYING[0] = False
    apply_widgets(rig)

    _APPLYING[0] = True
    try:
        s.controls.clear()
        s.groups.clear()
        s.limbs.clear()
        s.axes.clear()
        for key in _RESET:
            s.property_unset(key)
        s.initialised = False
    finally:
        _APPLYING[0] = False
    apply_all(rig)




class _RigOperator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)


class EFB_OT_rig_apply(_RigOperator):
    """Re-apply every setting to this rig. Keeps the pose and the action"""
    bl_idname = "efb.rig_apply"
    bl_label = "Apply Settings"

    def execute(self, context):
        apply_all(context.object)
        self.report({"INFO"}, "settings re-applied, pose untouched")
        return {"FINISHED"}


class EFB_OT_rig_reset_settings(_RigOperator):
    """Put every editable setting back to the value a freshly generated rig has"""
    bl_idname = "efb.rig_reset_settings"
    bl_label = "Reset Settings"

    def execute(self, context):
        reset_settings(context.object)
        return {"FINISHED"}


class EFB_OT_reset_widget(_RigOperator):
    """Put this control's widget back where the generator drew it"""
    bl_idname = "efb.reset_widget"
    bl_label = "Reset Widget"

    bone: StringProperty()
    everything: BoolProperty(default=False)

    def execute(self, context):
        rig = context.object
        s = ensure_rows(rig)
        rows = s.controls if self.everything else [r for r in s.controls
                                                   if r.bone == self.bone]
        _APPLYING[0] = True
        try:
            for row in rows:
                row.scale = 1.0
                row.offset = (0.0, 0.0, 0.0)
            if self.everything:
                s.widget_scale = 1.0
                for g in s.groups:
                    g.scale, g.offset = 1.0, 0.0
        finally:
            _APPLYING[0] = False
        apply_widgets(rig)
        return {"FINISHED"}


class EFB_OT_reveal_ik(_RigOperator):
    """Switch this limb to IK and select its IK handle, so the control is under the cursor
    rather than in a hidden collection"""
    bl_idname = "efb.reveal_ik"
    bl_label = "Find IK Control"

    limb: StringProperty(default="arm_r")

    def execute(self, context):
        rig = context.object
        limb = next((l for l in rig_limbs(rig) if l.key == self.limb), None)
        if limb is None:
            self.report({"ERROR"}, "no limb %r" % self.limb)
            return {"CANCELLED"}
        snap.set_mode(rig, limb, True, context)
        coll = rig.data.collections_all.get(limb.ik_collection)
        if coll is not None and not coll.is_visible_effectively:
            rig["show_inactive"] = True
            rig.update_tag()
            context.view_layer.update()
        for pb in rig.pose.bones:
            pb.select = False
        for name in (limb.ik_ctrl, limb.pole_ctrl):
            pb = rig.pose.bones.get(name)
            if pb is not None:
                pb.select = True
                pb.hide = pb.bone.hide = False
        rig.data.bones.active = rig.data.bones.get(limb.ik_ctrl)
        self.report({"INFO"}, "%s is on IK - %s selected" % (limb.label, limb.ik_ctrl))
        return {"FINISHED"}


class EFB_OT_setup(_RigOperator):
    """Read this rig and create its editable settings"""
    bl_idname = "efb.rig_setup"
    bl_label = "Set Up Editing"

    def execute(self, context):
        apply_all(context.object)
        return {"FINISHED"}


CLASSES = (EFB_PG_control, EFB_PG_group, EFB_PG_limb, EFB_PG_axis, EFB_PG_rig,
           EFB_OT_rig_apply,
           EFB_OT_rig_reset_settings, EFB_OT_reset_widget, EFB_OT_reveal_ik,
           EFB_OT_setup)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.efb_rig = PointerProperty(type=EFB_PG_rig)


def unregister():
    del bpy.types.Object.efb_rig
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
