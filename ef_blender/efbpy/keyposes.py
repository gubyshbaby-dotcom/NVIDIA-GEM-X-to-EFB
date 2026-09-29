"""Key poses: a dense clip turned into keys an animator can work in.

A capture - GEM-X, or any clip baked a key per frame - is correct and unworkable: every
channel carries a key on every frame, so moving a pose means moving twenty of them. What
an animator works with is a few keys on the poses that matter and smooth curves between.

So the dense curves are sampled once as the target, and the keys are rebuilt from scratch
as Blender's own BEZIER / AUTO_CLAMPED curves - the default an animator would get by hand.
A linear thinning would not do: a swinging arm curves between any two frames, and a line
needs a key every two or three of them to stay inside a degree, where a cubic needs one
every eight or nine. The fit starts from the turning points (a coarse efb.keyfit pass) and
then, measured on the real curves with FCurve.evaluate rather than on a model of them,
adds the worst frame of every segment that strays past the tolerance until none does.
That makes the tolerance a bound at every frame, not an average.

SHARED fits the whole rig at once and keys every bone on the same frames, so the dope
sheet reads as poses; PER_BONE lets each bone keep only its own. A channel that never
moves is left one key.

The Epic Fight export is unaffected in what it writes: its automatic sampling sees curves
the game cannot lerp and bakes them, so the game gets the smooth motion, densely.

Degrees bound each rotation; the Root's travel gets LOC_PER_DEGREE metres per degree.
Errors add down a chain, so a hand can stray by more than its own bone's share - at the
default two degrees a few centimetres at worst, under one Minecraft pixel.
"""

import math
import re

import bpy
from bpy.props import EnumProperty, FloatProperty

from efb.keyfit import fit_indices

from . import snap
from .animscene import action_curves

__all__ = ["SHARED", "PER_BONE", "EVERY_FRAME", "reduce_keys", "transform_curves",
           "MODES", "LOC_PER_DEGREE", "DEFAULT_MODE", "DEFAULT_DEGREES",
           "EFB_OT_reduce_keys", "register", "unregister"]

SHARED = "SHARED"
PER_BONE = "PER_BONE"
EVERY_FRAME = "ALL"

MODES = (
    (PER_BONE, "Per bone", "Each bone keeps only the keys it needs, on smooth curves. "
                           "The fewest keys - measured, a key every 4 frames on a brisk "
                           "walk and every 11 on everyday motion at 2 degrees"),
    (SHARED, "Key poses", "The same frames on every bone, so the dope sheet reads as "
                          "poses to grab and retime. Denser: whenever any bone needs a "
                          "key, every bone gets one"),
    (EVERY_FRAME, "Every frame", "A key per frame, linear. Exact, and heavy to edit"),
)

DEFAULT_MODE = PER_BONE

DEFAULT_DEGREES = 2.0

LOC_PER_DEGREE = 0.005

MAX_ROUNDS = 30

_PATH = re.compile(r'^pose\.bones\["(.+)"\]\.(location|rotation_quaternion|rotation_euler|'
                   r'scale)$')


def transform_curves(action, obj, bones=None) -> dict:
    """{bone: [fcurve, ...]} for the pose transform channels of `bones` (all when None)."""
    out = {}
    for fc in action_curves(action, obj):
        m = _PATH.match(fc.data_path)
        if not m or (bones is not None and m.group(1) not in bones):
            continue
        out.setdefault(m.group(1), []).append(fc)
    return out


def _channel(fc):
    return _PATH.match(fc.data_path).group(2)


def _tolerance(fc, degrees):
    kind = _channel(fc)
    if kind == "rotation_quaternion":
        return math.radians(degrees) / 4.0
    if kind == "rotation_euler":
        return math.radians(degrees) / 2.0
    if kind == "location":
        return degrees * LOC_PER_DEGREE / 2.0
    return degrees * 1e-3


def _write(fc, frames, values):
    """Replace a curve's keys with BEZIER / AUTO_CLAMPED ones at `frames`."""
    points = fc.keyframe_points
    try:
        points.clear()
    except AttributeError:
        for kp in reversed(list(points)):
            points.remove(kp, fast=True)
    points.add(len(frames))
    for kp, f, v in zip(points, frames, values):
        kp.co = (f, v)
        kp.interpolation = "BEZIER"
        kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
    fc.update()


def _fit(curves, frames, targets, tols):
    """Kept frame indices for one group of curves that share their keys."""
    n = len(frames)
    rows = [[targets[c][i] / tols[c] for c in range(len(curves))] for i in range(n)]
    keep = set(fit_indices(frames, rows, 3.0))
    for _round in range(MAX_ROUNDS):
        order = sorted(keep)
        for c, fc in enumerate(curves):
            _write(fc, [frames[i] for i in order], [targets[c][i] for i in order])
        err = [0.0] * n
        for c, fc in enumerate(curves):
            t, tol = targets[c], tols[c]
            for i in range(n):
                e = abs(fc.evaluate(frames[i]) - t[i]) / tol
                if e > err[i]:
                    err[i] = e
        added = False
        for a, b in zip(order, order[1:]):
            if b - a < 2:
                continue
            worst = max(range(a + 1, b), key=err.__getitem__)
            if err[worst] > 1.0:
                keep.add(worst)
                added = True
        if not added:
            break
    return sorted(keep)


def reduce_keys(obj, action=None, degrees=DEFAULT_DEGREES, mode=DEFAULT_MODE, bones=None,
                frame_range=None):
    """Rebuild `obj`'s transform keys as key poses. Returns (keys before, keys after,
    key poses or 0 for PER_BONE)."""
    ad = obj.animation_data
    action = action or (ad.action if ad else None)
    if action is None or degrees <= 0.0 or mode == EVERY_FRAME:
        return 0, 0, 0
    groups = transform_curves(action, obj, bones)
    curves = [fc for fcs in groups.values() for fc in fcs if len(fc.keyframe_points)]
    if not curves:
        return 0, 0, 0
    if frame_range is None:
        lo = min(fc.keyframe_points[0].co[0] for fc in curves)
        hi = max(fc.keyframe_points[-1].co[0] for fc in curves)
        frame_range = (int(math.floor(lo)), int(math.ceil(hi)))
    frames = [float(f) for f in range(frame_range[0], frame_range[1] + 1)]
    before = sum(len(fc.keyframe_points) for fc in curves)
    if len(frames) < 3:
        return before, before, 0

    targets = {fc: [fc.evaluate(f) for f in frames] for fc in curves}
    moving = [fc for fc in curves if max(targets[fc]) - min(targets[fc]) > 1e-7]
    for fc in curves:
        if fc not in moving:
            _write(fc, [frames[0]], [targets[fc][0]])

    units = [moving] if mode == SHARED else [
        [fc for fc in fcs if fc in moving] for fcs in groups.values()]
    poses = 0
    for unit in units:
        if not unit:
            continue
        keep = _fit(unit, frames, [targets[fc] for fc in unit],
                    [_tolerance(fc, degrees) for fc in unit])
        poses = max(poses, len(keep))
    after = sum(len(fc.keyframe_points) for fc in curves)
    obj.update_tag()
    return before, after, poses if mode == SHARED else 0


class EFB_OT_reduce_keys(bpy.types.Operator):
    """Rebuild the playing action's bone keys as few keys on smooth curves, inside a
    tolerance. It fits the curves as they are now, so a second pass stacks on the first"""
    bl_idname = "efb.reduce_keys"
    bl_label = "Reduce Keys"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(name="Keys", items=[m for m in MODES if m[0] != EVERY_FRAME],
                       default=DEFAULT_MODE)
    degrees: FloatProperty(name="Tolerance", default=DEFAULT_DEGREES, min=0.1, max=20.0,
                           description="How far, in degrees, a bone may stray from the "
                                       "curve it had, at any frame")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (snap.is_rig(obj) and obj.animation_data is not None
                and obj.animation_data.action is not None)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        rig = context.object
        before, after, poses = reduce_keys(rig, degrees=self.degrees, mode=self.mode,
                                           bones=_keyed_controls(rig))
        context.view_layer.update()
        self.report({"INFO"}, "%s: %d keys -> %d%s" % (
            rig.animation_data.action.name, before, after,
            " on %d key poses" % poses if poses else ""))
        return {"FINISHED"}


def _keyed_controls(rig):
    """Every bone but the IK side, whose keys are a bake the IK tools maintain."""
    from .animscene import IK_SIDE
    from .ops import rig_limbs

    ik = {getattr(l, f) for l in rig_limbs(rig) for f in IK_SIDE if getattr(l, f, "")}
    return {pb.name for pb in rig.pose.bones if pb.name not in ik}


CLASSES = (EFB_OT_reduce_keys,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
