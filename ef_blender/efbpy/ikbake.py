"""Put the IK handles where the FK pose already is, frame by frame.

An imported clip is FK keys - that is what Epic Fight stores - so the IK handles start at
rest and flipping a limb to IK throws the pose 0.22 to 1.14 m (measured on the seven
shipped subjects). Baking the handles from the FK result leaves both paths carrying the
same motion, which is what lets an animator switch a limb either way and keep working.

Only the IK side is keyed. The FK curves the import wrote are the file's own transforms and
stay untouched, so the limbs are left on FK and the re-export is still the file that came
in - efbpy.animscene.ik_controls is what keeps these keys out of it.

The dense walk is how the handles are found, not how they are left: `thin_ik_curves` drops
every sample the two keys around it already predict, which is what makes the result a
channel an animator can work in rather than a wall. On sword_auto1 that is 4884 keys down
to 1048, at a flip cost identical to the dense bake's to five decimals.

At import and not on demand, because the blend is a plain id property: the panel slider and
the gate both write it directly and nothing runs when they do, so a bake hung off a button
would leave a slider drag throwing the pose the 0.22 to 1.14 m it used to. Making the bake
unnecessary instead - parenting the handle under a mechanism bone that copies the FK end,
so its basis is a delta - would make IK additive over the FK curves, which is a different
rig and a different export.
"""

from __future__ import annotations

import bpy
from mathutils import Matrix

from efb.keyfit import constant, fit_indices, hemisphere_rows, normalised

from .animscene import IK_SIDE, action_curves, curve_bags
from .snap import snap_ik_to_fk

__all__ = ["bake_ik", "bake_frames", "baked_limbs", "limb_channels", "thin_ik_curves",
           "SLIDE_EPSILON", "KEY_TOLERANCE", "ROLE_TOLERANCE"]

_CHANNELS = ("location", "rotation_quaternion", "scale")

_KEYED = {"ik_upper": ("location",), "ik_ctrl": _CHANNELS, "pole_ctrl": _CHANNELS,
          "ik_twist": _CHANNELS}

SLIDE_EPSILON = 1e-9

KEY_TOLERANCE = 5e-5

ROLE_TOLERANCE = {"ik_upper": 1.0, "ik_ctrl": 1.0, "pole_ctrl": 8.0, "ik_twist": 16.0}


def limb_channels(obj, limb) -> list:
    """(bone, channels, tolerance multiple) for every IK-side bone the bake writes on."""
    return [(getattr(limb, f), _KEYED[f], ROLE_TOLERANCE[f]) for f in IK_SIDE
            if f in _KEYED and getattr(limb, f, "") and getattr(limb, f) in obj.pose.bones]


def bake_frames(frames=(), start=None, end=None) -> list:
    """Every whole frame across the clip, plus any key of its own that is not on one.

    Whole frames because that is where the clip is played and where FK's LINEAR segments
    and the IK solve are asked to agree; the clip's own keys because a rate that does not
    put a timestamp on a whole frame would otherwise leave that key unbaked.
    """
    got = {float(f) for f in frames}
    if not got and start is None:
        return []
    lo = int(min(got)) if start is None else int(start)
    hi = int(round(max(got))) if end is None else int(end)
    got.update(float(f) for f in range(lo, hi + 1))
    return sorted(f for f in got if lo <= f <= hi)


def bake_ik(obj, frames, limbs=None, context=None, tol=None) -> dict:
    """Key every limb's IK handle, pole, twist and chain root onto the FK pose.

    Ascending only. snap.py keys at frame - 1 to leave a clean hold, which going up lands
    on a frame already baked and going down overwrites a finished one - measured at
    0.324 m against 0.000274 m. Nothing here asks it to key, but the walk order is the
    same discipline, and the blend each snap leaves up is why every frame starts on FK.

    Returns {limb key: frames keyed}.
    """
    from .ops import rig_limbs

    context = context or bpy.context
    limbs = [l for l in (rig_limbs(obj) if limbs is None else limbs)
             if limb_channels(obj, l)]
    frames = sorted(frames)
    if not limbs or not frames:
        return {}

    scene = context.scene
    saved = scene.frame_current, scene.frame_subframe
    _drop_curves(obj, limbs)
    for frame in frames:
        whole = int(frame // 1)
        scene.frame_set(whole, subframe=float(frame - whole))
        _all_fk(obj, limbs, context)
        for limb in limbs:
            _seat_root(obj, limb, context)
            snap_ik_to_fk(obj, limb, context, keyframe=False, frame=frame)
            _key(obj, limb, frame)
    _all_fk(obj, limbs, context)
    scene.frame_set(saved[0], subframe=saved[1])
    _linear(obj, limbs)
    thin_ik_curves(obj, limbs, tol)
    return {l.key: len(frames) for l in limbs}


def thin_ik_curves(obj, limbs, tol=None) -> tuple:
    """Drop the baked keys the pair around them already predicts. Returns (before, after).

    Per bone rather than per curve: a limb whose channels keep the same key times reads as
    one column in the dope sheet, which is the shape an animator can grab. A channel that
    never moves is left one key, which costs nothing at all.
    """
    action = obj.animation_data.action if obj.animation_data else None
    tol = KEY_TOLERANCE if tol is None else tol
    if action is None or tol <= 0.0:
        return 0, 0
    before = after = 0
    for limb in limbs:
        for name, channels, multiple in limb_channels(obj, limb):
            was, now = _thin_bone(action, obj, name, channels, tol * multiple)
            before, after = before + was, after + now
    return before, after


def _thin_bone(action, obj, name, channels, tol) -> tuple:
    pairs = _bone_curves(action, obj, name, channels)
    if not pairs:
        return 0, 0
    counts = {len(fc.keyframe_points) for _, fc in pairs}
    before = sum(len(fc.keyframe_points) for _, fc in pairs)
    if len(counts) != 1 or before == 0:
        return before, before
    n = counts.pop()
    times = [float(kp.co[0]) for kp in pairs[0][1].keyframe_points]
    rows = [[float(fc.keyframe_points[i].co[1]) for _, fc in pairs] for i in range(n)]
    spans = [(i, 4) for i, (chan, fc) in enumerate(pairs)
             if chan == "rotation_quaternion" and fc.array_index == 0]
    rows = hemisphere_rows(rows, spans)

    moving = [c for c in range(len(pairs))
              if not constant([row[c] for row in rows])]
    if n < 3 or not moving:
        keep = [0] if not moving else list(range(n))
    else:
        sub = [[row[c] for c in moving] for row in rows]
        keep = fit_indices(times, normalised(sub, [tol] * len(moving)), 1.0)

    plan, after = [], 0
    for c, (chan, fc) in enumerate(pairs):
        want = [0] if c not in moving else keep
        after += len(want)
        plan.append((fc.data_path, fc.array_index, fc.group.name if fc.group else "",
                     [(times[i], rows[i][c]) for i in want]))
    _rewrite(action, obj, plan)
    return before, after


def _bone_curves(action, obj, name, channels) -> list:
    """(channel, fcurve) for one bone, in a fixed channel then axis order."""
    escaped = bpy.utils.escape_identifier(name)
    paths = {'pose.bones["%s"].%s' % (escaped, c): c for c in channels}
    out = [(paths[fc.data_path], fc) for fc in action_curves(action, obj)
           if fc.data_path in paths]
    out.sort(key=lambda p: (channels.index(p[0]), p[1].array_index))
    return out


def _rewrite(action, obj, plan):
    """Replace a bone's curves with the keys given. Rebuilt rather than trimmed in place -
    removing a keyframe at a time is quadratic and a 700-frame clip pays for it."""
    wanted = {(path, index) for path, index, _g, _k in plan}
    for bag in curve_bags(action, obj):
        for fc in [f for f in bag.fcurves if (f.data_path, f.array_index) in wanted]:
            bag.fcurves.remove(fc)
    bag = next(iter(curve_bags(action, obj)), None)
    if bag is None:
        return
    for path, index, group, keys in plan:
        fc = bag.fcurves.new(path, index=index)
        if group:
            if group not in {g.name for g in bag.groups}:
                bag.groups.new(group)
            fc.group = bag.groups[group]
        fc.keyframe_points.add(len(keys))
        for k, (frame, value) in enumerate(keys):
            kp = fc.keyframe_points[k]
            kp.co = (frame, value)
            kp.interpolation = "LINEAR"
        fc.update()


def _paths(obj, limbs) -> set:
    out = set()
    for limb in limbs:
        for name, channels, _mult in limb_channels(obj, limb):
            escaped = bpy.utils.escape_identifier(name)
            out.update('pose.bones["%s"].%s' % (escaped, c) for c in channels)
    return out


def _drop_curves(obj, limbs):
    """A bake replaces what is on the IK side; it does not add to it.

    Baking a second time over a shorter range would otherwise leave the old keys standing
    past the end of the new one, and the limb would solve to a pose nothing asked for.
    """
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return
    paths = _paths(obj, limbs)
    for bag in curve_bags(action, obj):
        for fc in [f for f in bag.fcurves if f.data_path in paths]:
            bag.fcurves.remove(fc)


def baked_limbs(obj, action=None) -> list:
    """The limbs whose IK handle carries keys in this action - what a bake left behind."""
    from .ops import rig_limbs

    action = action or (obj.animation_data.action if obj.animation_data else None)
    if action is None:
        return []
    keyed = {fc.data_path for fc in action_curves(action, obj)}
    return [l for l in rig_limbs(obj)
            if 'pose.bones["%s"].location'
            % bpy.utils.escape_identifier(l.ik_ctrl) in keyed]


def _seat_root(obj, limb, context):
    """Slide the solver chain's root onto the joint the clip actually put there.

    Epic Fight keys a translation on every joint and two of the seven subjects slide a hip
    - 0.038 m in sword_dash, 0.050 m in greatsword_airslash - while the solver chain hangs
    off the rest offset from its parent and has nowhere to put that. Left alone it costs
    the whole limb: 0.083 m at Knee_R.
    """
    if not limb.ik_upper or limb.ik_upper not in obj.pose.bones:
        return
    ik, fk = obj.pose.bones[limb.ik_upper], obj.pose.bones[limb.fk_upper]
    delta = fk.matrix.translation - ik.matrix.translation
    if delta.length < SLIDE_EPSILON:
        return
    ik.matrix = Matrix.Translation(delta) @ ik.matrix
    obj.update_tag()
    context.view_layer.update()


def _all_fk(obj, limbs, context):
    for limb in limbs:
        obj[limb.prop] = 0.0
    obj.update_tag()
    context.view_layer.update()


def _key(obj, limb, frame):
    for name, channels, _mult in limb_channels(obj, limb):
        pb = obj.pose.bones[name]
        for channel in channels:
            pb.keyframe_insert(channel, frame=frame)


def _linear(obj, limbs):
    """Epic Fight lerps, and so does the FK side these keys have to agree with; a bezier
    handle between two baked frames would leave the two paths apart in between."""
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return
    paths = _paths(obj, limbs)
    for fc in action_curves(action, obj):
        if fc.data_path not in paths:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.update()
