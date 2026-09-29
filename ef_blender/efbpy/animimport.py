"""Epic Fight animation json -> a Blender action on the rig.

This is the half of the round trip an artist actually looks at: if the rest matrices or
the basis are wrong, an imported walk cycle looks wrong immediately, long before any
test says so.

The keys land on the pose channels directly - location, rotation_quaternion, scale -
because that is what pose_bone.matrix_basis is made of and what Epic Fight stores in a
TransformSheet. Interpolation is set to LINEAR: Epic Fight lerps translation and scale
and nlerps rotation, so bezier handles in Blender would describe a curve the game cannot
reproduce.

Those keys are FK, and on their own they leave the IK handles at rest, so flipping a limb
to IK threw the pose up to 1.14 m. efbpy.ikbake walks the same frames afterwards and puts
the handles where the FK result is, which is what makes the switch safe either way.
"""

from __future__ import annotations

import json
import math

import bpy
from mathutils import Matrix, Quaternion

from efb import clip as efc
from efb.animjson import AnimationDocument, loads
from efb.rig import COORD_BONE, Rig

from .animscene import (CLIP_PROP, action_curves, find_rig, fk_controls,
                        offset_location_keys, seam_controls, seam_shift)
from .restpose import local_offset

__all__ = ["import_document", "import_file", "import_from_jar", "META_PROP",
           "read_meta", "clip_name", "parked_clips", "clip_range"]

META_PROP = CLIP_PROP

_CHANNELS = (("location", 3), ("rotation_quaternion", 4), ("scale", 3))
_IDENTITY = Matrix.Identity(4)


def import_document(obj, rig: Rig, doc: AnimationDocument, *, fps=None,
                    name="clip", source="", set_scene=True, action=None, bake_ik=True):
    """Key `doc` onto `obj` as an action. Returns (action, ImportedClip)."""
    fps = fps or efc.document_fps(doc)
    imported = efc.read_document(rig, doc, fps, source=source)
    _stand_the_pin_down(obj, imported)

    action = action or bpy.data.actions.new(name)
    if obj.animation_data is None:
        obj.animation_data_create()
    obj.animation_data.action = action
    _assign_slot(obj, action)

    targets = fk_controls(obj)
    _force_fk(obj)

    for bone in rig.bones():
        target = targets.get(bone, bone)
        if target not in obj.pose.bones:
            continue
        frames = imported.frames.get(bone) or []
        channels = imported.channels.get(bone) or []
        pb = obj.pose.bones[target]
        pb.rotation_mode = "QUATERNION"
        off = local_offset(pb)
        if not frames:
            pb.matrix_basis = _IDENTITY if off is None else off.inverted()
            continue
        if off is not None:
            channels = _undo_override(channels, off)
        path = 'pose.bones["%s"].' % bpy.utils.escape_identifier(target)
        for ci, (prop, count) in enumerate(_CHANNELS):
            for axis in range(count):
                fc = _curve(action, path + prop, axis, target)
                fc.keyframe_points.add(len(frames))
                for k, frame in enumerate(frames):
                    kp = fc.keyframe_points[k]
                    kp.co = (frame, channels[k][ci][axis])
                    kp.interpolation = "LINEAR"
                fc.update()

    _store_meta(action, imported.meta)
    if set_scene:
        _set_range(imported, fps)
    _compensate_seams(obj, action, imported, targets)
    if bake_ik:
        from .ikbake import bake_frames, bake_ik as run_bake
        start, end = clip_range(imported)
        run_bake(obj, bake_frames(_clip_frames(imported), start, end))
    return action, imported


def rolled_joints(obj, imported) -> dict:
    """Pinned lower joints this clip rolls or folds sideways, and by how much.

    In degrees off the fold axis, from the rotation vector rather than a euler, because a
    euler puts the whole rotation on X near the fold's own gimbal and would read a pure
    roll as no roll at all.
    """
    from efb.rigedit import PIN_EPSILON, hinges
    from efbpy.ops import rig_limbs
    from efbpy.rigedit import settings

    s = settings(obj)
    if s is None:
        return {}
    out = {}
    for h in hinges(rig_limbs(obj)):
        row = s.limb(h.limb)
        if row is None or not (s.limits and row.limits and row.pin):
            continue
        worst = 0.0
        for _loc, quat, _scale in imported.channels.get(h.deform) or ():
            axis, angle = Quaternion(quat).normalized().to_axis_angle()
            if angle > math.pi:
                angle, axis = 2.0 * math.pi - angle, -axis
            v = axis * math.degrees(angle)
            worst = max(worst, math.hypot(v.y, v.z))
        if worst > PIN_EPSILON:
            out[h.deform] = worst
    return out


def parked_clips(obj) -> list:
    """Epic Fight clips this rig carries on NLA tracks. Replay is a rig setting rather
    than an action one, so lowering it would change how these play back."""
    ad = obj.animation_data
    return [st.action for t in (ad.nla_tracks if ad else ())
            for st in t.strips if st.action is not None and st.action.get(CLIP_PROP)]


def _stand_the_pin_down(obj, imported):
    """A clip that rolls a forearm or a shin cannot be replayed through the pin - the
    exporter reads the posed bone, so the pin would rewrite the file on its way back out.
    Raise Replay instead and leave the animator to decide, rather than clamping quietly.

    Replay tracks the clip that is loaded, so a hinge-only clip puts it back down - but
    only while the rig has no parked clip that needs it up. Lowering it under a stashed
    clip would clamp that clip's shins behind its back, which the animator never asked for.
    """
    from efbpy.rigedit import settings

    s = settings(obj)
    if s is None:
        return
    rolled = rolled_joints(obj, imported)
    if bool(rolled) != s.replay and (rolled or not parked_clips(obj)):
        s.replay = bool(rolled)
    if not rolled and s.replay:
        print("efb: Replay stays up - %d parked clip(s) need it" % len(parked_clips(obj)))
    if rolled:
        print("efb: %s roll off the fold axis (worst %.1f deg) - Replay raised so the "
              "clip exports as it arrived" % (", ".join(sorted(rolled)),
                                              max(rolled.values())))


def _undo_override(channels, off):
    """Take a rest override back out of a clip's keys.

    The keys are absolute Epic Fight joint transforms, so an imported clip has to land on
    the pose the file describes and not on top of whatever the animator made the new zero.
    Signs are carried key to key: decomposing each one alone can flip the quaternion and
    Blender would interpolate the long way round.
    """
    inv = off.inverted()
    out, prev = [], None
    for loc, quat, scale in channels:
        m = inv @ (Matrix.Translation(loc) @ Quaternion(quat).to_matrix().to_4x4()
                   @ Matrix.Diagonal(tuple(scale) + (1.0,)))
        t, q, s = m.decompose()
        if prev is not None and q.dot(prev) < 0.0:
            q.negate()
        prev = q
        out.append((tuple(t), tuple(q), tuple(s)))
    return out


def _compensate_seams(obj, action, imported, targets):
    """Epic Fight slides Elbow_*/Knee_* itself, so an imported clip already carries the
    seam translation in its marker tracks and the rig's seam constraint would lay a second
    copy on top. Take the constraint's own contribution back out of the marker's keys.

    Measured off the live rig at each key rather than predicted from the map, so it is
    exact whatever the constraint is set to, and the re-export is the file that came in.
    """
    pairs = {m: c for m, c in seam_controls(obj).items() if imported.frames.get(m)}
    frames = {f for m in pairs for f in imported.frames[m]}
    if not pairs:
        return
    shift = seam_shift(obj, pairs, frames)
    offset_location_keys(obj, action, pairs,
                         {m: {f: -v for f, v in per.items()}
                          for m, per in shift.items()})


def _force_fk(obj):
    """An imported clip is FK keys; leaving a limb on IK would let the solver overrule
    them. Writing an id property does not tag the depsgraph, hence update_tag."""
    from .ops import rig_limbs
    changed = False
    for limb in rig_limbs(obj):
        if obj.get(limb.prop):
            obj[limb.prop] = 0.0
            changed = True
    if changed:
        obj.update_tag()


def import_file(path, armature, obj=None, *, coord=None, **kw):
    with open(path, "rb") as fh:
        doc = loads(fh.read())
    return _import_onto(doc, armature, obj, coord, source=str(path), **kw)


def import_from_jar(jar, clip_name, obj=None, *, coord=None, **kw):
    """Import straight out of the mod jar - the fastest way to check the rig."""
    doc = jar.clip(clip_name)
    armature = jar.armature_for(clip_name)
    return _import_onto(doc, armature, obj, coord, source=clip_name, **kw)


def _import_onto(doc, armature, obj, coord, **kw):
    obj = find_rig(obj)
    if coord is None:
        coord = COORD_BONE in obj.pose.bones
    rig = Rig(armature, coord=coord)
    kw.setdefault("name", clip_name(kw.get("source", "clip")))
    return import_document(obj, rig, doc, **kw)


def clip_name(source):
    """The action name a clip gets: its file stem. Blender numbers a repeat itself, so
    importing the same clip twice gives walk and walk.001."""
    base = str(source).replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".json"):
        base = base[:-5]
    return base or "clip"


def _curve(action, data_path, index, group):
    for fc in action_curves(action):
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    return _new_curve(action, data_path, index, group)


def _new_curve(action, data_path, index, group):
    if hasattr(action, "layers"):
        bag = _channelbag(action)
        fc = bag.fcurves.new(data_path, index=index)
        if group not in {g.name for g in bag.groups}:
            bag.groups.new(group)
        fc.group = bag.groups[group]
        return fc
    fc = action.fcurves.new(data_path, index=index, action_group=group)
    return fc


def _channelbag(action):
    if not action.slots:
        action.slots.new("OBJECT", "rig")
    if not action.layers:
        action.layers.new("base")
    layer = action.layers[0]
    if not layer.strips:
        layer.strips.new(type="KEYFRAME")
    strip = layer.strips[0]
    bag = strip.channelbag(action.slots[0])
    return bag or strip.channelbag(action.slots[0], ensure=True)


def _assign_slot(obj, action):
    if not hasattr(action, "slots"):
        return
    if not action.slots:
        _channelbag(action)
    try:
        obj.animation_data.action_slot = action.slots[0]
    except (AttributeError, TypeError):
        pass


def _clip_frames(imported):
    return [f for frames in imported.frames.values() for f in frames]


def clip_range(imported):
    """(start, end) - the scene range this clip plays over, or None when it keys nothing."""
    got = _clip_frames(imported)
    if not got:
        return None, None
    return int(min(got)), max(int(min(got)) + 1, int(round(max(got))))


def _set_range(imported, fps):
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / float(fps)
    start, end = clip_range(imported)
    if start is not None:
        scene.frame_start, scene.frame_end = start, end


def _store_meta(action, meta: efc.ClipMeta):
    action[META_PROP] = json.dumps({
        "fps": meta.fps,
        "format": meta.format,
        "order": list(meta.order),
        "key_order": list(meta.key_order),
        "extras": meta.extras,
        "foreign": {n: t.to_json() for n, t in meta.foreign.items()},
        "camera": meta.camera.to_json() if meta.camera is not None else None,
        "source": meta.source,
    })


def read_meta(action) -> "efc.ClipMeta | None":
    """The Epic Fight sidecar an import left on the action, if any."""
    raw = action.get(META_PROP) if action else None
    if not raw:
        return None
    from efb.animjson import CameraTrack, Track
    obj = json.loads(raw)
    fmt = obj.get("format")
    default = "attributes" if fmt == "attributes" else "matrix"
    return efc.ClipMeta(
        fps=obj.get("fps", 60.0), format=fmt,
        order=tuple(obj.get("order", ())), key_order=tuple(obj.get("key_order", ())),
        extras=obj.get("extras", {}),
        foreign={n: Track.from_json(t, default)
                 for n, t in (obj.get("foreign") or {}).items()},
        camera=CameraTrack.from_json(obj["camera"]) if obj.get("camera") else None,
        source=obj.get("source", ""))
