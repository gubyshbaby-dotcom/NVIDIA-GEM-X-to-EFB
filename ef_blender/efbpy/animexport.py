"""A Blender action -> Epic Fight animation json.

Sampling policy, and why the default is what it is:

  auto (default)   export keyed, measure what Epic Fight's playback of those keys does
                   against the Blender pose on a dense ladder, and bake instead when it
                   drifts past efb.clip.AUTO_DRIFT. Costs one extra scene walk and
                   nothing else: an action Epic Fight can already reproduce stays keyed
                   and comes out byte for byte what `keyed` would have written.

  keyed            every bone emits its own keyframes and nothing else. This is what
                   upstream did - 339 of the 383 shipped clips give different joints
                   different time arrays, and biped/living/walk.json is 7 authored keys
                   over 0.783 s, not a 47-frame bake. Exact only when the curves are
                   LINEAR and nothing between the keys is solved rather than stored.

  baked            one shared frame ladder at `step` frames. Bezier or constant
                   interpolation, constraints, IK, drivers, NLA. Epic Fight only lerps
                   translation and scale and nlerps rotation between the keys it is
                   handed, so anything curved has to be sampled.

`info["drift"]` is measured at the frames that were exported; `info["lerp_drift"]` is the
same thing on the dense ladder, which is the number that says whether the shape survived.
They differ exactly when the keys are sparse, which is the case worth knowing about.

Which joints get a track is the separate `coverage` axis - see efb.clip.export_joints.
"""

from __future__ import annotations

import bpy

from efb import clip as efc
from efb.animjson import dumps
from efb.rig import COORD_BONE, Rig

from .animimport import read_meta
from .animscene import (action_curves, bind_switches, detach_controls, find_rig,
                        fk_controls, from_matrix, ik_controls, inert_ik_bones,
                        is_constrained, seam_followers, unbound_tools)

__all__ = ["sample_action", "export_action", "export_file", "action_keyframes",
           "action_of", "pose_delta"]

_PREFIX = "pose.bones["


def action_of(obj):
    ad = obj.animation_data
    return ad.action if ad else None


_curves = action_curves


def action_keyframes(action, obj=None) -> dict:
    """bone name -> the sorted set of frames anything on that bone is keyed at."""
    out = {}
    for fc in _curves(action, obj):
        path = fc.data_path
        if not path.startswith(_PREFIX):
            continue
        end = path.find('"]', len(_PREFIX) + 1)
        if end < 0:
            continue
        bone = path[len(_PREFIX) + 1:end]
        got = out.setdefault(bone, set())
        for kp in fc.keyframe_points:
            got.add(float(kp.co[0]))
    return {k: sorted(v) for k, v in out.items()}


def pose_delta(pb):
    """Epic Fight's TransformSheet value read off a bone something else is driving.

    matrix_basis is what the animator typed before constraints, which on a constrained
    deform bone is always identity, so the pose has to come back out of the evaluated
    matrices. Stripping with the bone's own Blender rest rather than the json one is what
    keeps this equal to matrix_basis: the json rest goes back on in build_document, and
    Tool_R/Tool_L's stray scale would otherwise be lost on the way through and back.
    """
    pose = from_matrix(pb.matrix)
    rest = from_matrix(pb.bone.matrix_local)
    if pb.parent is not None:
        pose = from_matrix(pb.parent.matrix).inverse() @ pose
        rest = from_matrix(pb.parent.bone.matrix_local).inverse() @ rest
    return rest.inverse() @ pose


def _remap_keys(keys, controls, ik=None, follows=None, chains=None, switches=None):
    """A control rig carries its keys on the controls; tracks are named after the deform
    bones they drive, so fold each control's frames onto the joints it moves. One FK
    control drives one joint; one IK target drives the whole chain under it.

    A seam marker nobody keyed rides the bone it seams, or the slide the constraint gives
    it never reaches the file. A marker that carries keys of its own - which is every
    imported clip, all 20 joints - keeps them, or the export invents timestamps the
    original never had.

    `chains` is the unbound tool: its delta is a counter-animation of its parents, so it
    needs a key wherever any of them has one whether or not it has keys of its own.

    `switches` is the keyed bind. The socket's world rule changes at those frames and
    nothing else in the file records that, so they go in even when nobody keyed the
    socket or its parents there.
    """
    back = {ctrl: [joint] for joint, ctrl in controls.items()}
    for ctrl, joints in (ik or {}).items():
        back.setdefault(ctrl, []).extend(joints)
    out = {}
    for bone, frames in keys.items():
        for joint in back.get(bone, (bone,)):
            out.setdefault(joint, set()).update(frames)
    for lower, markers in (follows or {}).items():
        for marker in markers:
            if not out.get(marker) and out.get(lower):
                out[marker] = set(out[lower])
    for joint, above in (chains or {}).items():
        got = set(out.get(joint) or ())
        for parent in above:
            got.update(out.get(parent) or ())
        if got:
            out[joint] = got
    for joint, frames in (switches or {}).items():
        out.setdefault(joint, set()).update(frames)
    return {k: sorted(v) for k, v in out.items()}


def sample_action(obj, rig: Rig, *, mode=efc.KEYED, step=1.0, frame_range=None,
                  bones=None, action=None, constrained=None,
                  coverage=efc.JOINTS_ALL, source_order=()) -> efc.PoseTable:
    """Walk the frames and read the pose. The only part that needs a scene."""
    action = action or action_of(obj)
    keys = action_keyframes(action, obj) if action else {}
    controls = fk_controls(obj)
    ik = ik_controls(obj, action)
    for detach, joints in detach_controls(obj).items():
        ik.setdefault(detach, []).extend(joints)
    for name in inert_ik_bones(obj, action):
        keys.pop(name, None)
    follows = seam_followers(obj)
    tools = unbound_tools(obj, action)
    switches = bind_switches(obj, action)
    if controls or ik:
        keys = _remap_keys(keys, controls, ik, follows, tools, switches)
    names = (list(bones) if bones is not None
             else efc.export_joints(rig, keys, coverage, source_order))
    if constrained is None:
        constrained = is_constrained(obj, names)
    frame_range = frame_range or _range(keys, action, obj)
    keyed = efc.plan_frames(mode, keys, frame_range, step=step, bones=names)

    frames = sorted({f for got in keyed.values() for f in got})
    table = efc.PoseTable(frames=frames,
                          basis={n: [] for n in names},
                          world={n: [] for n in names},
                          channels={n: [] for n in names},
                          keyed=keyed)
    scene = bpy.context.scene
    saved = scene.frame_current, scene.frame_subframe
    for f in frames:
        whole = int(f // 1)
        scene.frame_set(whole, subframe=float(f - whole))
        for n in names:
            pb = obj.pose.bones[n]
            table.world[n].append(from_matrix(pb.matrix))
            if constrained:
                delta = pose_delta(pb)
                table.basis[n].append(delta)
                table.channels[n].append(delta.decompose())
                continue
            table.basis[n].append(from_matrix(pb.matrix_basis))
            table.channels[n].append(
                (tuple(pb.location), tuple(pb.rotation_quaternion), tuple(pb.scale))
                if pb.rotation_mode == "QUATERNION"
                else table.basis[n][-1].decompose())
    scene.frame_set(saved[0], subframe=saved[1])
    return table


def _range(keys, action, obj):
    got = [f for frames in keys.values() for f in frames]
    if got:
        return (min(got), max(got))
    if action is not None and hasattr(action, "frame_range"):
        return tuple(action.frame_range)
    scene = bpy.context.scene
    return (float(scene.frame_start), float(scene.frame_end))


def export_action(obj=None, armature=None, *, mode=efc.AUTO, step=1.0,
                  frame_range=None, fps=None, format=None, order=None, bones=None,
                  meta=None, action=None, report_drift=True, constrained=None,
                  coverage=efc.JOINTS_ALL, drift_limit=efc.AUTO_DRIFT):
    """Returns (AnimationDocument, info dict). `armature` may be omitted when the action
    carries an import's sidecar and the rig was built from the same json."""
    obj = find_rig(obj)
    action = action or action_of(obj)
    if action is None:
        raise RuntimeError("%s has no action" % obj.name)
    meta = meta if meta is not None else read_meta(action)
    if armature is None:
        raise RuntimeError("export_action needs the Epic Fight armature json")
    rig = Rig(armature, coord=COORD_BONE in obj.pose.bones)
    if fps is None:
        fps = meta.fps if meta else _scene_fps()

    walk = dict(step=step, frame_range=frame_range, bones=bones, action=action,
                constrained=constrained, coverage=coverage,
                source_order=meta.order if meta else ())
    chosen = efc.KEYED if mode == efc.AUTO else mode
    table = sample_action(obj, rig, mode=chosen, **walk)
    doc = efc.build_document(rig, table, fps, order=order, format=format, meta=meta)

    lerp_drift = lerp_at = None
    if mode == efc.AUTO or (report_drift and chosen == efc.KEYED):
        dense = sample_action(obj, rig, mode=efc.BAKED, **walk)
        lerp_drift, lerp_at = efc.interpolation_error(doc, rig, dense, fps)
        if mode == efc.AUTO and lerp_drift > drift_limit:
            chosen, table = efc.BAKED, dense
            doc = efc.build_document(rig, table, fps, order=order, format=format,
                                     meta=meta)
            lerp_drift, lerp_at = efc.interpolation_error(doc, rig, table, fps)

    info = {"fps": fps, "mode": chosen, "asked": mode, "frames": len(table.frames),
            "coverage": coverage, "tracks": len(doc.tracks),
            "keys": sum(len(t) for t in doc.tracks)}
    if lerp_drift is not None:
        info["lerp_drift"], info["lerp_drift_at"] = lerp_drift, lerp_at
    if report_drift:
        info["drift"], info["drift_at"] = efc.interpolation_error(doc, rig, table, fps)
    return doc, info


def export_file(path, obj=None, armature=None, **kw):
    doc, info = export_action(obj, armature, **kw)
    with open(path, "wb") as fh:
        fh.write(dumps(doc))
    info["path"] = str(path)
    return doc, info


def _scene_fps():
    r = bpy.context.scene.render
    return r.fps / r.fps_base
