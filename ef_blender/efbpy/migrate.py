"""Bringing an action forward from an older Epic Fight rig.

The route is the Epic Fight document, not a bake and not a channel copy. The exporter
reads the 20 deform joints and discovers the wiring live, so it runs unchanged on a rig
six versions old; the importer writes those joints onto whatever rig is in front of it.
A channel copy looks free - every control name still exists, nothing errors - and lands
centimetres out, because seven of the animator-facing controls were reparented under
their old names.

Everything here reads the old rig and writes only the new one.
"""

from __future__ import annotations

import os

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty

from efb import assets, bundled
from efb import clip as efc
from efb.rig import COORD_BONE, Rig
from efb.rigdef import ADDON_VERSION, RIG_ID
from efb.rigedit import seams
from efb.rigver import DEFORM_ONLY, VERSION_MARKS, ladder_version, mark_for

from . import snap
from .animexport import export_action
from .animimport import import_document
from .animscene import ARMATURE_PROP, action_curves, to_matrix
from .ops import rig_limbs

__all__ = ["VERSION_MARKS", "DEFORM_ONLY", "LOSSLESS", "SNAPPED", "MIGRATION_PROP",
           "Detected", "Report", "ladder_version", "detect", "resolve_armature",
           "rest_gap", "deform_gap", "migrate_action", "whole_frames", "candidates",
           "append_rigs", "draw_dialog", "NO_CANDIDATE", "OLD_RIG_COLLECTION",
           "register", "unregister", "CLASSES"]

REST_TOLERANCE = 5e-3

LOSSLESS = 1e-5
SNAPPED = 1e-3

OLD_POLE_PARENT = "CTRL-Master"

MIGRATION_PROP = "efb_migration"

NO_CANDIDATE = "__none__"

OLD_RIG_COLLECTION = "EF Old Rig"

_SNAP_MODES = (("AUTO", "As the old rig had them",
                "Snap back only the limbs the old action had on IK, over the frames it "
                "had them there"),
               ("ALL", "Every limb", "Snap all four limbs back onto IK"),
               ("NONE", "Leave on FK", "Leave the result on FK, which is what the bake "
                                       "produces"))




class Detected:
    """What a rig turned out to be. `stamp` is None on a rig that carries no
    rig_version, which is every file older than the stamp itself."""

    def __init__(self, version, stamp, entity, armature, rest, mark=""):
        self.version = version
        self.stamp = stamp
        self.entity = entity
        self.armature = armature
        self.rest = rest
        self.mark = mark

    @property
    def agrees(self):
        return self.stamp is None or self.stamp == self.version

    def __str__(self):
        name = "deform only" if self.version == DEFORM_ONLY else "version %d" % self.version
        return "%s, %s build" % (name, self.entity)


def _pose_bones(obj):
    """An armature object that has never been evaluated has no pose yet."""
    return getattr(obj.pose, "bones", ()) if obj.pose is not None else ()


def rest_gap(obj, rig: Rig) -> float:
    """Worst element between this object's deform bones and the armature json's rests.
    inf when a joint is missing, which is what refuses a foreign rig."""
    worst = 0.0
    for name in rig.deform_bones():
        bone = obj.data.bones.get(name)
        if bone is None or not bone.use_deform:
            return float("inf")
        want = to_matrix(rig.rest_model(name))
        got = bone.matrix_local
        worst = max(worst, max(abs(got[r][c] - want[r][c])
                               for r in range(4) for c in range(4)))
    return worst


_ARMATURES = {}


def _armature(entity, jar):
    key = (entity, jar or "")
    if key not in _ARMATURES:
        _ARMATURES[key] = assets.armature(entity, jar)
    return _ARMATURES[key]


def resolve_armature(obj, jar=None):
    """(entity, Armature, rest gap) for the build whose rests this rig actually carries.

    An old rig may name no build at all - the two pack rigs carry no efb_armature - and
    the two builds differ by half a texel at the arms, so the rests decide and the name is
    only a hint about which to try first.
    """
    named = obj.get(ARMATURE_PROP)
    order = ([named] if named in bundled.VARIANTS else []) + \
            [e for e in sorted(bundled.VARIANTS) if e != named]
    coord = COORD_BONE in _pose_bones(obj)
    best = None
    for entity in order:
        try:
            arm = _armature(entity, jar)
        except Exception:
            continue
        gap = rest_gap(obj, Rig(arm, coord=coord))
        if best is None or gap < best[2]:
            best = (entity, arm, gap)
    if best is None:
        raise RuntimeError("no Epic Fight armature available to read %s against" % obj.name)
    return best


def detect(obj, jar=None) -> Detected:
    """What version of this add-on's rig `obj` is. Raises on anything that is not ours."""
    if obj is None:
        raise RuntimeError("nothing to read: pick the old rig in the list, or use "
                           "Append Old Rig if it lives in another .blend")
    if obj.type != "ARMATURE":
        raise RuntimeError("%s is a %s, not an armature. The old rig is the armature "
                           "object itself, not the body it drives"
                           % (obj.name, obj.type.lower()))
    entity, armature, gap = resolve_armature(obj, jar)
    if gap == float("inf"):
        raise RuntimeError("%s has no Epic Fight deform joints, so there is nothing here "
                           "to bring across. Use Append Old Rig to bring the old rig in "
                           "from the .blend the shot was animated in" % obj.name)
    if gap > REST_TOLERANCE:
        raise RuntimeError("%s carries the 20 Epic Fight joint names but rests %.3f m away "
                           "from them, so it is somebody else's Epic Fight rig and not an "
                           "older one of ours. Export its animation as an Epic Fight json "
                           "and bring that in with File > Import > Epic Fight Animation"
                           % (obj.name, gap))
    stamp = obj.data.get("rig_version")
    names = [b.name for b in obj.data.bones]
    parents = {b.name: (b.parent.name if b.parent else None) for b in obj.data.bones}
    if obj.data.get("rig_id") != RIG_ID:
        if ladder_version(names, obj.keys(), parents) is not None:
            raise RuntimeError("%s has this rig's control bones but no rig_id, so it was "
                               "built by hand or by a tree mid-change. Generate a rig, "
                               "migrate onto that, and keep this file untouched" % obj.name)
        return Detected(DEFORM_ONLY, stamp, entity, armature, gap, "the 20 joints alone")
    version = ladder_version(names, obj.keys(), parents)
    if version is None:
        raise RuntimeError("%s is one of ours but carries no version mark this build "
                           "knows, so it is newer than this add-on. Update the add-on and "
                           "run this again" % obj.name)
    return Detected(version, stamp, entity, armature, gap, mark_for(version))




def _quiet(obj, jar=None):
    try:
        return detect(obj, jar), ""
    except RuntimeError as exc:
        return None, str(exc)


def _readable(context, jar=None):
    """[(object, Detected)] for the rigs in play, and the active object.

    The selection first, because that is what the animator pointed at. One rig selected is
    the ordinary case though, and it says nothing about where a migration should land, so
    the search then widens to the whole file - which is where the old rig actually is.
    """
    context = context or bpy.context
    active = getattr(context, "object", None)
    pool = list(getattr(context, "selected_objects", ()) or ()) + \
           ([active] if active is not None else [])

    def rigs_in(objects):
        out, seen = [], set()
        for obj in objects:
            if obj is None or obj.type != "ARMATURE" or obj.name in seen:
                continue
            seen.add(obj.name)
            got = _quiet(obj, jar)[0]
            if got is not None:
                out.append((obj, got))
        return out

    out = rigs_in(pool)
    return (out if len(out) > 1 else rigs_in(list(pool) + list(bpy.data.objects))), active


def migration_target(context=None, jar=None):
    """Which rig a migration writes onto: the newest of the ones in play.

    The button reads as something done TO the old rig, so it is pressed with either rig
    active. This only ever carries actions forward, so the newest one is the only possible
    destination - choosing it beats refusing and telling the animator to reselect.
    """
    rigs, active = _readable(context, jar)
    if not rigs:
        return active
    rigs.sort(key=lambda row: (-row[1].version, row[0] is not active, row[0].name))
    return rigs[0][0]


def destination(context=None, source=None, jar=None):
    """Which rig a migration writes onto once the old one is known.

    The active object, unless that IS the old rig - which is what happens when the button
    is pressed with the old rig picked in the viewport. Then it is the newest rig in play
    that is at least as new as the source; None when there is no such rig, so the caller
    still reaches the refusal that says where an old rig comes from.
    """
    context = context or bpy.context
    rigs, active = _readable(context, jar)
    if source is None:
        return migration_target(context, jar)
    if active is not None and active is not source:
        return active
    mine = next((d for o, d in rigs if o is source), None)
    rest = [row for row in rigs
            if row[0] is not source and (mine is None or row[1].version >= mine.version)]
    if not rest:
        return None
    rest.sort(key=lambda row: (-row[1].version, row[0].name))
    return rest[0][0]


def candidates(context, target=None, jar=None):
    """(usable, refused) armatures in this file, for the rig `target` is migrating onto.

    Usable is [(object, Detected)], selected first, so pressing the button with the old rig
    selected fills the field with it. Refused is [(name, why)] and is what the dialog shows
    instead of an empty list - an animator whose old rig is in another .blend has to be told
    that, not handed a blank field.
    """
    context = context or bpy.context
    target = target or migration_target(context, jar)
    onto = _quiet(target, jar)[0] if target is not None else None
    selected = set(getattr(context, "selected_objects", ()) or ())
    usable, refused = [], []
    for obj in bpy.data.objects:
        if obj is target or obj.type != "ARMATURE":
            continue
        got, why = _quiet(obj, jar)
        if got is None:
            refused.append((obj.name, why))
        elif onto is not None and onto.version < got.version:
            refused.append((obj.name, "is newer than %s (version %d against %d), and this "
                                      "carries actions forward only"
                            % (target.name, got.version, onto.version)))
        else:
            usable.append((obj, got))
    usable.sort(key=lambda row: (row[0] not in selected, -row[1].version, row[0].name))
    return usable, refused


def append_rigs(path, context=None, with_actions=True, jar=None):
    """Append every Epic Fight rig in `path` into this file and return them.

    Appended, never linked: the migration assigns actions on the old rig as it walks them,
    and a linked object refuses every write. Whatever else the file holds is loaded to be
    looked at and then thrown away again, because a .blend index does not say which object
    is an armature until it is in memory.
    """
    context = context or bpy.context
    if not os.path.isfile(path):
        raise RuntimeError("no such file: %s" % path)
    before = set(bpy.data.objects)
    actions_before = set(bpy.data.actions)
    try:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
            if with_actions:
                dst.actions = list(src.actions)
    except Exception as exc:
        raise RuntimeError("%s could not be read as a .blend: %s" % (path, exc))
    loaded = [o for o in bpy.data.objects if o not in before and o is not None]
    rigs, why = [], []
    for obj in loaded:
        if obj.type != "ARMATURE":
            continue
        got, reason = _quiet(obj, jar)
        (rigs if got is not None else why).append(obj if got is not None else reason)
    for obj in loaded:
        if obj not in rigs:
            bpy.data.objects.remove(obj, do_unlink=True)
    if not rigs:
        for action in [a for a in bpy.data.actions if a not in actions_before]:
            bpy.data.actions.remove(action)
        _purge()
        name = os.path.basename(path)
        if why:
            raise RuntimeError("nothing in %s can be migrated from. %s" % (name, why[0]))
        raise RuntimeError("%s holds no armature at all. Open it and check the shot was "
                           "animated in that file" % name)
    group = bpy.data.collections.get(OLD_RIG_COLLECTION)
    if group is None:
        group = bpy.data.collections.new(OLD_RIG_COLLECTION)
    if group.name not in context.scene.collection.children:
        context.scene.collection.children.link(group)
    for obj in rigs:
        if obj.name not in group.objects:
            group.objects.link(obj)
        obj.hide_render = True
    _purge()
    return rigs


def _purge():
    """The meshes and materials that came in behind the rigs and were dropped again."""
    try:
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)
    except (AttributeError, TypeError, RuntimeError):
        pass




def deform_gap(old, new, frames, joints, context=None):
    """(worst head metres, where, worst element) between two rigs playing the same range.

    This is exactly what the exporter reads, so it is the whole of what a migration has to
    get right. Both matrices are in their own object's space, so the spawn yaw cancels.
    """
    scene = (context or bpy.context).scene
    saved = scene.frame_current, scene.frame_subframe
    worst, elem, where = 0.0, 0.0, ""
    for frame in frames:
        whole = int(frame // 1)
        scene.frame_set(whole, subframe=float(frame - whole))
        for name in joints:
            a, b = old.pose.bones.get(name), new.pose.bones.get(name)
            if a is None or b is None:
                continue
            head = (a.matrix.translation - b.matrix.translation).length
            if head > worst:
                worst, where = head, "%s@%g" % (name, frame)
            elem = max(elem, max(abs(a.matrix[r][c] - b.matrix[r][c])
                                 for r in range(4) for c in range(4)))
    scene.frame_set(saved[0], subframe=saved[1])
    return worst, where, elem




class Report:
    """What happened, in the three bands the animator has to act on. Never a bare success
    message: "could not come across" is never empty, and every approximation carries its
    number."""

    def __init__(self, old, new, detected, current):
        self.old, self.new = old.name, new.name
        self.detected, self.current = detected, current
        self.actions = []
        self.bands = {"transferred": [], "approximated": [], "lost": []}
        self.worst, self.worst_at, self.frames = 0.0, "", 0
        self.bake_worst, self.bake_at = 0.0, ""
        self.failed = False

    def add(self, band, text):
        self.bands[band].append(text)

    def bake(self, worst, where):
        if worst > self.bake_worst:
            self.bake_worst, self.bake_at = worst, where

    def score(self, name, made, frames, worst, where):
        self.actions.append((name, made, frames, worst))
        self.frames += frames
        if worst > self.worst:
            self.worst, self.worst_at = worst, where

    @property
    def verdict(self):
        if self.worst <= LOSSLESS:
            return "lossless"
        if self.worst <= SNAPPED:
            return "within the snap-back band"
        return "FAILED"

    def headline(self):
        return ("%s -> %s: %d action(s), worst joint %.6f m over %d frames (%s)"
                % (self.old, self.new, len(self.actions), self.worst, self.frames,
                   self.verdict))

    def text(self):
        d = self.detected
        stamp = "no rig_version at all" if d.stamp is None else "stamped %d" % d.stamp
        agree = "" if d.agrees else "   <- they disagree; trust the detection"
        out = ["Migrated %d action(s) from %s onto %s" % (len(self.actions), self.old,
                                                          self.new),
               "  old rig  %s, %s%s" % (d, stamp, agree),
               "  new rig  version %d, add-on %s" % (self.current, ADDON_VERSION),
               "  detected by  %s" % d.mark]
        for name, made, frames, worst in self.actions:
            out.append("  %-24s -> %-28s %3d frames  %.6f m" % (name, made, frames, worst))
        for band, title in (("transferred", "Transferred exactly"),
                            ("approximated", "Approximated"),
                            ("lost", "Could not come across")):
            out.append("")
            out.append(title)
            for line in self.bands[band] or ["(nothing)"]:
                out.append("  " + line)
        out += ["", "%s has not been touched." % self.old]
        return "\n".join(out)




def _limb_track(obj, action, prop):
    """{frame: value} for one limb's IK blend on the old rig, or {} when it is not keyed.
    Read off the curve because that is what the rig actually played."""
    if action is None:
        return {}
    path = '["%s"]' % prop
    for fc in action_curves(action, obj):
        if fc.data_path == path:
            return {round(float(kp.co[0]), 3): float(kp.co[1])
                    for kp in fc.keyframe_points}
    return {}


def _ik_frames(old, action, prop, frames):
    """The frames of `frames` this limb was on IK at, evaluating the old curve rather than
    the property, so a limb that switches mid-clip comes across as it was."""
    track = _limb_track(old, action, prop)
    if not track:
        return list(frames) if float(old.get(prop, 0.0)) >= 0.5 else []
    times = sorted(track)
    out = []
    for f in frames:
        prev = [t for t in times if t <= f]
        value = track[prev[-1]] if prev else track[times[0]]
        if value >= 0.5:
            out.append(f)
    return out


def _action_range(action, obj):
    if action is not None and hasattr(action, "frame_range"):
        lo, hi = action.frame_range
        if hi > lo:
            return float(lo), float(hi)
    scene = bpy.context.scene
    return float(scene.frame_start), float(scene.frame_end)


def _curve_stats(action, obj):
    curves = action_curves(action, obj) if action else []
    return len(curves), sum(len(fc.keyframe_points) for fc in curves)


def _compensate(old, new, detected, report):
    """Settings on the NEW rig that reproduce how the old one behaved.

    The fold flip was the one entry and version 10 has no such switch: the fold side is
    the solver hinge's, one-sided, so there is nothing to carry. Flipped arms come back
    through the snap oracle instead, which walks both sides and keeps the closer one.
    """
    if detected.version == DEFORM_ONLY:
        return
    flipped = [k for k in ("arm_r", "arm_l") if old.get("fold_flip_" + k, 0)]
    if flipped:
        report.add("lost",
                   "fold side: %s were on the flipped side and this rig has no such "
                   "switch - the hinge names one side. The snap tries both and keeps the "
                   "closer one; check the elbows" % " and ".join(flipped))


def _set_limits(new, on):
    """Turn this rig's Joint Limits on or off. False when it was already there."""
    from . import rigedit
    s = rigedit.settings(new)
    if s is None or bool(s.limits) == bool(on):
        return False
    s.limits = bool(on)
    rigedit.apply_all(new)
    return True


def _relax_limits(new, old, frames, joints, bake, bake_at, context, report):
    """The one repair the oracle can drive: a limb the old rig had on IK can end up past
    this rig's own hinge stop, and the FK control it lands on is clamped there.

    Measured: an arm whose pole was on the far side reproduced at 0.2177 m with Joint
    Limits on and 0.000001 m with them off. Only tried when the bake did not land, and
    put straight back when it was not the cause.
    """
    if bake <= LOSSLESS or not _set_limits(new, False):
        return bake, bake_at
    context.view_layer.update()
    relaxed, where, _ = deform_gap(old, new, frames, joints, context)
    if relaxed >= bake:
        _set_limits(new, True)
        context.view_layer.update()
        return bake, bake_at
    report.add("approximated",
               "this rig's Joint Limits were clamping the old pose (%.4f m against "
               "%.6f m without them), so they are now OFF. Turn them back on from the "
               "sidebar once you have looked at the elbows and wrists" % (bake, relaxed))
    return relaxed, where


def _lost_lines(old, new, detected, report, doc_joints):
    version = detected.version
    if version == DEFORM_ONLY:
        report.add("lost", "the old rig had no controls at all, so there was no IK, no "
                           "pole and no setting to bring across - only the 20 joints")
    elif version < 5:
        report.add("lost",
                   "pole space: the old rig hung its poles off %s and this rig has no "
                   "such space. Left at the default (Chord). Check the knees on any clip "
                   "that travels" % OLD_POLE_PARENT)
    report.add("lost", "curve shape: bezier and constant interpolation are gone, "
                       "everywhere. The bake is one linear key per frame")
    old_props = set(old.keys())
    new_props = set(new.keys())
    fresh = len(new_props - old_props)
    if fresh:
        report.add("lost", "settings: %d of this rig's %d properties did not exist on the "
                           "old one and are at their defaults" % (fresh, len(new_props)))
    report.add("lost", "widgets: control sizes, offsets and any rest override on the old "
                       "rig were not carried")
    missing = [n for n in doc_joints if n not in new.pose.bones]
    if missing:
        report.add("lost", "joints this rig has no bone for: %s" % ", ".join(missing))


def migrate_action(new, old, *, actions=None, snap_back="AUTO", jar=None, context=None,
                   name_suffix="_migrated"):
    """Carry `old`'s actions onto `new` through the Epic Fight document. Returns a Report.

    `new` is only ever written and `old` only ever read - its action assignment is put back
    the way it was found.
    """
    context = context or bpy.context
    if old is new:
        raise RuntimeError("%s is the rig being migrated ONTO, so it cannot also be the "
                           "one read from. The old rig is a second armature: bring it in "
                           "with Append Old Rig, then pick it from the list" % new.name)
    detected = detect(old, jar)
    current = detect(new, jar)
    if current.version == DEFORM_ONLY:
        raise RuntimeError("%s is a plain deform rig with no controls to write onto. "
                           "Generate a rig from the Build panel, make it the active "
                           "object, and run this from there" % new.name)
    if current.version < detected.version:
        raise RuntimeError("%s is OLDER than %s (version %d against %d). Make the newer "
                           "rig the active object - this button only carries actions "
                           "forward" % (new.name, old.name, current.version,
                                        detected.version))

    report = Report(old, new, detected, current.version)
    if not detected.agrees:
        report.add("approximated",
                   "the file stamps rig_version %d but is structurally version %d. It was "
                   "built from a tree mid-change; the detection is what was used"
                   % (detected.stamp, detected.version))
    if detected.entity != current.entity:
        report.add("approximated",
                   "the old rig is the %s build and this one is %s: the arms differ by "
                   "half a texel, so the pose lands on this rig's own rests"
                   % (detected.entity, current.entity))

    ad = old.animation_data
    was = ad.action if ad else None
    todo = list(actions) if actions is not None else ([was] if was else [])
    if not todo:
        raise RuntimeError("%s has no action assigned. Give it the action you want carried "
                           "over in the Action editor, or tick Every action to take every "
                           "clip in the file it can play" % old.name)

    _compensate(old, new, detected, report)

    coord = COORD_BONE in old.pose.bones
    rig = Rig(detected.armature, coord=coord)
    joints = rig.deform_bones()
    before, after, snapped = (0, 0), (0, 0), []
    try:
        for action in todo:
            if ad is None:
                old.animation_data_create()
                ad = old.animation_data
            ad.action = action
            context.view_layer.update()
            start, end = _action_range(action, old)
            frames = [float(f) for f in whole_frames(start, end)]
            before = tuple(a + b for a, b in zip(before, _curve_stats(action, old)))
            doc, _ = export_action(old, detected.armature, mode=efc.BAKED, step=1.0,
                                   frame_range=(start, end), action=action,
                                   coverage=efc.JOINTS_ALL, report_drift=False)
            fresh = bpy.data.actions.new(action.name + name_suffix)
            import_document(new, rig, doc, name=fresh.name, source=action.name,
                            set_scene=False, action=fresh, bake_ik=False)
            after = tuple(a + b for a, b in zip(after, _curve_stats(fresh, new)))
            _assign(new, fresh)
            context.view_layer.update()
            bake, bake_at, _ = deform_gap(old, new, frames, joints, context)
            bake, bake_at = _relax_limits(new, old, frames, joints, bake, bake_at,
                                          context, report)
            report.bake(bake, bake_at)
            got = []
            if snap_back != "NONE":
                fresh, got = _snap_back(new, old, action, fresh, frames, snap_back,
                                        context, report, joints)
            snapped += got
            if got:
                worst, where, _ = deform_gap(old, new, frames, joints, context)
            else:
                worst, where = bake, bake_at
            report.score(action.name, fresh.name, len(frames), worst, where)
    finally:
        if ad is not None:
            ad.action = was

    report.add("transferred", "%d Epic Fight joints over %d frames, one document per "
                              "action. The export reads these joints and nothing else"
               % (len(joints), report.frames))
    report.add("transferred", "worst joint head straight out of the bake %.6f m at %s"
               % (report.bake_worst, report.bake_at or "-"))
    report.add("approximated",
               "curves are one key per frame, linear: %d channels / %d keys became "
               "%d / %d" % (before[0], before[1], after[0], after[1]))
    if seams(new.pose.bones.keys()):
        report.add("transferred", "the seam slide on the elbow and knee markers was taken "
                                  "back out of the imported keys, measured off this rig")
    if snapped:
        report.add("approximated", "worst joint head after snapping %.6f m at %s"
                   % (report.worst, report.worst_at or "-"))
    report.failed = (report.bake_worst > LOSSLESS
                     or (snapped and report.worst > SNAPPED))
    if report.failed:
        report.add("lost", "VERIFICATION FAILED: %.6f m is past the %.0e m this route "
                           "lands at. Keep the old rig"
                   % (report.worst, SNAPPED if snapped else LOSSLESS))
    _lost_lines(old, new, detected, report, joints)
    if not snapped:
        report.add("lost", "IK handles carry no keys: the bake puts every limb on FK. "
                           "Snap a limb back to IK from the sidebar when you need one")
    for _, made, _, _ in report.actions:
        action = bpy.data.actions.get(made)
        if action is not None:
            action[MIGRATION_PROP] = report.text()
    return report


def whole_frames(start, end):
    """The bake's own ladder: every whole frame in the range, which is what step 1.0
    produces."""
    lo, hi = int(round(start)), int(round(end))
    return list(range(lo, hi + 1))


def _assign(obj, action):
    if obj.animation_data is None:
        obj.animation_data_create()
    obj.animation_data.action = action
    if hasattr(action, "slots") and action.slots:
        try:
            obj.animation_data.action_slot = action.slots[0]
        except (AttributeError, TypeError):
            pass


def _fold_options(new, limb):
    """The fold sides worth trying, best first.

    The fold-flip switch is gone with version 10 - the solver hinge names one side - so
    the only remaining lever is the pole space, and flipping that would trade a measured
    0.82 percent wrong-side rate for 29.37 behind the animator's back. One try.
    """
    return "", (None,)


def _limb_bones(new, limb):
    """Everything a snap writes on this limb - the same list snap.py keys."""
    names = [limb.fk_upper, limb.fk_lower, limb.ik_ctrl, limb.pole_ctrl]
    return [new.pose.bones[n] for n in names if n and n in new.pose.bones]


def _capture(new, limb):
    return (float(new.get(limb.prop, 0.0)),
            {pb.name: pb.matrix_basis.copy() for pb in _limb_bones(new, limb)})


def _restore(new, limb, state, context):
    """Put a rejected snap back. The blend is an id property and the handles carry no keys
    in the bake, so without this a thrown-away trial leaves the limb solving on IK."""
    value, basis = state
    for name, matrix in basis.items():
        new.pose.bones[name].matrix_basis = matrix
    new[limb.prop] = value
    new.update_tag()
    context.view_layer.update()


def _walk_snap(new, limb, want, context):
    """Ascending only: snap.py keys at frame - 1 to leave a clean hold, which lands on an
    already-snapped frame going up and overwrites a finished one going down. Measured,
    descending costs 0.32 m against 0.00027 m."""
    for frame in sorted(want):
        context.scene.frame_set(int(frame))
        snap.snap_ik_to_fk(new, limb, context, keyframe=True, frame=int(frame))


def _snap_back(new, old, old_action, fresh, frames, mode, context, report, joints):
    """Put the limbs the old action had on IK back onto IK, one limb at a time, keeping
    each only if the 20 joints still agree. Returns (the action to keep, limbs snapped).

    A snap that cannot reproduce the old fold is thrown away rather than shipped: the
    limb stays on the bake's FK keys, which are exact, and the report says so.
    """
    plan = {}
    for limb in rig_limbs(new):
        want = (list(frames) if mode == "ALL"
                else _ik_frames(old, old_action, limb.prop, frames))
        if want:
            plan[limb.key] = (limb, sorted(set(want)))
    if not plan:
        return fresh, []

    saved_frame = context.scene.frame_current
    working, kept = fresh, []
    for key in sorted(plan):
        limb, want = plan[key]
        prop, tries = _fold_options(new, limb)
        first = tries[0]
        state = _capture(new, limb)
        best = None
        for value in tries:
            trial = working.copy()
            if prop:
                new[prop] = value
                new.update_tag()
            _assign(new, trial)
            context.view_layer.update()
            _walk_snap(new, limb, want, context)
            worst, where, _ = deform_gap(old, new, frames, joints, context)
            if best is None or worst < best[1]:
                if best is not None:
                    bpy.data.actions.remove(best[0])
                best = (trial, worst, where, value)
            else:
                bpy.data.actions.remove(trial)
            if worst <= SNAPPED:
                break
        trial, worst, where, value = best
        if worst <= SNAPPED:
            if working is not fresh:
                bpy.data.actions.remove(working)
            working = trial
            if prop:
                new[prop] = value
            kept.append(limb)
            report.add("approximated", "%s snapped back onto IK over %d of %d frames%s"
                       % (limb.label, len(want), len(frames),
                          ", fold side flipped to reach it" if prop and value != first
                          else ""))
        else:
            bpy.data.actions.remove(trial)
            if prop:
                new[prop] = first
            _assign(new, working)
            _restore(new, limb, state, context)
            report.add("lost", "%s could not go back on IK: the nearest this rig's solver "
                               "reached was %.4f m at %s, so it is left on the bake's FK "
                               "keys, which are exact" % (limb.label, worst, where))
        new.update_tag()
        _assign(new, working)
        context.view_layer.update()
    context.scene.frame_set(saved_frame)
    if working is not fresh:
        name = fresh.name
        bpy.data.actions.remove(fresh)
        working.name = name
        _assign(new, working)
        context.view_layer.update()
    if kept:
        report.add("approximated", "the snap writes a holding key one frame before each "
                                   "stretch it takes over, so the action starts a frame "
                                   "early")
    return working, kept




_ITEMS = []


def _old_rig_items(self, context):
    _ITEMS.clear()
    for obj, got in candidates(context)[0]:
        _ITEMS.append((obj.name, "%s  (%s)" % (obj.name, got),
                       "Read the action off %s, %s, add-on stamp %s"
                       % (obj.name, got, obj.data.get("rig_addon", "none"))))
    if not _ITEMS:
        _ITEMS.append((NO_CANDIDATE, "No older rig in this file",
                       "Append the old rig from its own .blend first"))
    return list(_ITEMS)


def draw_dialog(layout, op, context):
    """The dialog body. A function so the gate can run it without a window."""
    usable, refused = candidates(context)
    layout.use_property_split = True
    if usable:
        layout.prop(op, "old_pick")
        layout.prop(op, "snap_back")
        layout.prop(op, "every_action")
    else:
        col = layout.column(align=True)
        col.label(text="No other Epic Fight rig is in this file.", icon="INFO")
        col.label(text="Close this, press Append Old Rig above, and")
        col.label(text="choose the .blend the old shot was animated in.")
        col.label(text="The old rig lands in a collection you can delete after.")
    if refused:
        box = layout.column(align=True)
        box.scale_y = 0.8
        box.label(text="Not offered:", icon="ERROR")
        for name, why in refused[:3]:
            box.label(text="  %s - %s" % (name, why[:90]))


class EFB_OT_migrate_append(bpy.types.Operator):
    """Append the old Epic Fight rig, with its actions, out of another .blend so it can be
    migrated from. Nothing in that file is written"""
    bl_idname = "efb.migrate_append_rig"
    bl_label = "Append Old Rig..."
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.blend", options={"HIDDEN"})
    with_actions: BoolProperty(name="Its actions too", default=True,
                               description="Bring every action in that file across as "
                                           "well, not only the one the rig has assigned")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            rigs = append_rigs(self.filepath, context, self.with_actions)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in rigs:
            obj.select_set(True)
        self.report({"INFO"}, "appended %s into '%s' - now press Migrate Action"
                    % (", ".join(o.name for o in rigs), OLD_RIG_COLLECTION))
        return {"FINISHED"}


class EFB_OT_migrate_action(bpy.types.Operator):
    """Carry the actions off an older Epic Fight rig onto this one, through the Epic Fight
    document. The old rig is only read"""
    bl_idname = "efb.migrate_action"
    bl_label = "Migrate Action From Old Rig"
    bl_options = {"REGISTER", "UNDO"}

    old_pick: EnumProperty(name="Old rig", items=_old_rig_items,
                           description="Which armature in this file to take the action off")
    old_rig: StringProperty(name="Old rig by name", default="",
                            description="Names the old rig for a script. The button uses "
                                        "the list instead", options={"SKIP_SAVE"})
    snap_back: EnumProperty(name="Limbs", items=_SNAP_MODES, default="AUTO")
    every_action: BoolProperty(name="Every action", default=False,
                               description="Migrate every action in the file that the old "
                                           "rig can play, not only the one assigned")

    @classmethod
    def poll(cls, context):
        return snap.is_rig(context.object)

    def invoke(self, context, event):
        self.old_rig = ""
        usable = candidates(context)[0]
        if usable:
            self.old_pick = usable[0][0].name
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        draw_dialog(self.layout, self, context)

    def _chosen(self, context):
        """The old rig, from a script's name first and the picker second."""
        if self.old_rig:
            named = bpy.data.objects.get(self.old_rig)
            if named is None:
                raise RuntimeError("this file has no object called %r" % self.old_rig)
            return named
        if self.old_pick and self.old_pick != NO_CANDIDATE:
            picked = bpy.data.objects.get(self.old_pick)
            if picked is not None:
                return picked
        raise RuntimeError("no older Epic Fight rig is in this file yet. Press Append Old "
                           "Rig and choose the .blend the old shot was animated in")

    def execute(self, context):
        new = context.object
        try:
            old = self._chosen(context)
            if not self.old_rig:
                new = destination(context, old) or new
            actions = _playable(old) if self.every_action else None
            report = migrate_action(new, old, actions=actions,
                                    snap_back=self.snap_back, context=context)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        print(report.text())
        if new is not context.object:
            self.report({"INFO"}, "%s was the newer rig, so the action was written onto it"
                        % new.name)
        for band in ("approximated", "lost"):
            for line in report.bands[band]:
                self.report({"WARNING"}, "%s: %s" % (band, line))
        self.report({"ERROR"} if report.failed else {"INFO"}, report.headline())
        return {"FINISHED"}


def _playable(old):
    """Every action in the file that keys a bone this rig has. An old file usually holds
    one per shot and none of them is assigned."""
    names = set(old.pose.bones.keys())
    out = []
    for action in bpy.data.actions:
        if MIGRATION_PROP in action:
            continue
        for fc in action_curves(action, None):
            path = fc.data_path
            if path.startswith('pose.bones["'):
                end = path.find('"]', 12)
                if end > 0 and path[12:end] in names:
                    out.append(action)
                    break
    return out or None


CLASSES = (EFB_OT_migrate_append, EFB_OT_migrate_action)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
