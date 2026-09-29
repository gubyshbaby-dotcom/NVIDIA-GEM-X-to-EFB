"""Headless gate: GEM-X import onto the real control rig, in a real Blender.

    blender -b --factory-startup --python-exit-code 1 --python tests/blender_gemx.py -- \
        path/to/hpe_results.pt [--render out_dir] [--frames 0,30,60]

Checks that the operator spawns the rig, keys the clip, leaves every limb on FK with the
IK handles baked, and that the deform bones Blender evaluates are the pose the retarget
wrote - the same comparison the Epic Fight export makes - to 1e-4. Then that the default
key reduction stays inside its tolerance on every frame and that "Key poses" lines every
bone up; and that switching the rig wide -> slim lands on exactly the rig Generate makes
for slim, keeps every key and every pose delta, and switches back. With --render it also
writes workbench PNGs of the listed frames.
"""

import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "ef_blender"))

import efbpy  # noqa: E402
from efb import bundled  # noqa: E402
from efb.clip import pose_table_from_clip, read_document  # noqa: E402
from efb.gemx import RetargetOptions, retarget  # noqa: E402
from efb.rig import Rig  # noqa: E402
from efb.soma import load_motion  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"path": argv[0], "render": None, "frames": None}
    i = 1
    while i < len(argv):
        if argv[i] == "--render":
            out["render"] = argv[i + 1]
            i += 2
        elif argv[i] == "--frames":
            out["frames"] = [int(v) for v in argv[i + 1].split(",")]
            i += 2
        else:
            i += 1
    return out


def fail(msg):
    print("GATE FAIL:", msg)
    sys.exit(1)


def main():
    a = args()
    efbpy.register()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

    res = bpy.ops.efb.import_gemx(filepath=a["path"], keys="ALL")
    if res != {"FINISHED"}:
        fail("operator returned %s" % res)
    rig = bpy.context.view_layer.objects.active
    if rig is None or rig.type != "ARMATURE":
        fail("no rig spawned")
    action = rig.animation_data.action
    print("rig", rig.name, "bones", len(rig.data.bones), "action", action.name,
          "range", bpy.context.scene.frame_start, bpy.context.scene.frame_end,
          "fps", bpy.context.scene.render.fps)
    for limb in ("ik_arm_r", "ik_arm_l", "ik_leg_r", "ik_leg_l"):
        if rig.get(limb, 0.0) != 0.0:
            fail("%s left on IK" % limb)
    s = getattr(rig, "efb_rig", None)
    print("replay raised:", getattr(s, "replay", None), " joint limits:",
          getattr(s, "limits", None))
    from efbpy.ikbake import baked_limbs
    if len(baked_limbs(rig, action)) != 4:
        fail("IK baked on %d limbs, not 4" % len(baked_limbs(rig, action)))

    # What the retarget wrote, evaluated without Blender...
    arm = bundled.armature(rig.get("efb_armature", "biped"))
    motion = load_motion(a["path"], fps=30.0)
    doc, info = retarget(motion, arm, RetargetOptions())
    rig_def = Rig(arm, coord=False)
    clip = read_document(rig_def, doc, info.fps_out)
    table = pose_table_from_clip(rig_def, clip)

    # ...against what Blender evaluates on the deform bones.
    # The seam markers are left to the rig, which slides them itself. The Tool sockets
    # ride a Child Of holder whose seated inverse costs about half a millimetre - the
    # rig's own, measured the same on any clip - so they get their own bound.
    scene = bpy.context.scene
    frames = a["frames"] or sorted({0, len(table.frames) // 3, 2 * len(table.frames) // 3,
                                    len(table.frames) - 1})
    bounds = {"body": (1e-4, 0.0, ""), "tool": (2e-3, 0.0, "")}
    for f in range(len(table.frames)):
        scene.frame_set(int(table.frames[f]))
        for name in rig_def.deform_bones():
            if name.startswith(("Knee", "Elbow")):
                continue
            kind = "tool" if name.startswith("Tool") else "body"
            want = table.world[name][f]
            got = rig.pose.bones[name].matrix
            d = max(abs(got[r][c] - want[r, c]) for r in range(3) for c in range(4))
            tol, worst, _ = bounds[kind]
            if d > worst:
                bounds[kind] = (tol, d, "%s@%d" % (name, f))
    for kind, (tol, worst, where) in bounds.items():
        print("deform %s pose vs retarget: worst %.2e at %s" % (kind, worst, where))
        if worst > tol:
            fail("rig does not play the clip back (%.4g at %s)" % (worst, where))

    feet = []
    for f in range(len(table.frames)):
        scene.frame_set(int(table.frames[f]))
        for leg in ("Leg_R", "Leg_L"):
            pb = rig.pose.bones[leg]
            feet.append((rig.matrix_world @ pb.tail).z)
    print("foot height: min %.4f  2%%-quantile %.4f  max %.4f"
          % (min(feet), sorted(feet)[len(feet) // 50], max(feet)))

    ik_flip_holds(rig, table)
    export_matches(a["path"], arm)
    keys_hold(a["path"], arm)
    switch_holds(a["path"])

    if a["render"]:
        render(rig, a["render"], [int(table.frames[f]) for f in frames])
    print("GATE OK")


def ik_flip_holds(rig, table):
    """Flipping any limb to IK anywhere in the clip keeps the pose: the bake put the
    handles where the capture is. Measured at the limb's tip, in metres."""
    from efbpy.ops import rig_limbs

    scene = bpy.context.scene
    step = max(1, len(table.frames) // 60)
    worst, where = 0.0, ""
    for limb in rig_limbs(rig):
        for i in range(0, len(table.frames), step):
            scene.frame_set(int(table.frames[i]))
            fk = rig.pose.bones[limb.lower].tail.copy()
            rig[limb.prop] = 1.0
            rig.update_tag()
            bpy.context.view_layer.update()
            d = (rig.pose.bones[limb.lower].tail - fk).length
            rig[limb.prop] = 0.0
            rig.update_tag()
            bpy.context.view_layer.update()
            if d > worst:
                worst, where = d, "%s@%d" % (limb.key, i)
    print("IK flip moves a limb tip by at most %.4f m (%s)" % (worst, where))
    if worst > 0.01:
        fail("switching to IK moves the pose %.3f m at %s" % (worst, where))


def _fresh():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)


def keys_hold(path, arm):
    """Default import (per bone, 2 degrees): every FK control within 2 degrees of the
    capture on every frame, the Root within its travel bound; then "Key poses" keys every
    bone on the same frames."""
    from efb import quat as Q
    from efbpy.animscene import action_curves, fk_controls
    from efbpy.keyposes import LOC_PER_DEGREE

    dense, info = retarget(load_motion(path, fps=30.0), arm, RetargetOptions())
    for mode in ("PER_BONE", "SHARED"):
        _fresh()
        if bpy.ops.efb.import_gemx(filepath=path, keys=mode, key_tolerance=2.0) \
                != {"FINISHED"}:
            fail("import with keys=%s failed" % mode)
        rig = bpy.context.view_layer.objects.active
        action = rig.animation_data.action
        ctrl = fk_controls(rig)
        curves = {}
        for fc in action_curves(action, rig):
            curves[(fc.data_path, fc.array_index)] = fc
        worst_rot = worst_loc = 0.0
        frames_of = set()
        total = 0
        for track in dense.tracks:
            name = ctrl.get(track.name, track.name)
            base = 'pose.bones["%s"].' % name
            quat = [curves.get((base + "rotation_quaternion", i)) for i in range(4)]
            if None in quat:
                continue
            frames_of.add(tuple(round(k.co[0], 3) for k in quat[1].keyframe_points))
            total += sum(len(fc.keyframe_points) for fc in quat)
            loc = [curves.get((base + "location", i)) for i in range(3)]
            for f, key in enumerate(track.keys):
                got = tuple(fc.evaluate(f) for fc in quat)
                worst_rot = max(worst_rot, math.degrees(Q.angle(got, key.rot)))
                if track.name == "Root" and None not in loc:
                    worst_loc = max(worst_loc, max(abs(fc.evaluate(f) - v)
                                                   for fc, v in zip(loc, key.loc)))
        print("keys=%s: %d rotation keys over %d frames, worst %.3f deg, root %.4f m, "
              "%d distinct key-frame sets" % (mode, total, info.frames_out, worst_rot,
                                               worst_loc, len(frames_of)))
        if worst_rot > 2.0 + 1e-3 or worst_loc > 2.0 * LOC_PER_DEGREE:
            fail("keys=%s strays past its tolerance" % mode)
        if mode == "SHARED" and len(frames_of) != 1:
            fail("key poses do not line the bones up")


def switch_holds(path):
    """Wide rig with a clip -> Slim -> Wide. The slim result is compared against a fresh
    Generate of slim, bone by bone, and the clip against itself."""
    from efbpy.animscene import action_curves, local_delta
    from efbpy.body import bodies_of

    _fresh()
    bpy.ops.efb.import_gemx(filepath=path, keys="PER_BONE", variant="biped")
    rig = bpy.context.view_layer.objects.active
    action = rig.animation_data.action
    keys = {(fc.data_path, fc.array_index): [tuple(k.co) for k in fc.keyframe_points]
            for fc in action_curves(action, rig) if "IK-" not in fc.data_path
            and "POLE-" not in fc.data_path and ".ik" not in fc.data_path
            and ".twist" not in fc.data_path}
    scene = bpy.context.scene
    probe = [0, scene.frame_end // 2, scene.frame_end]
    deform = [b.name for b in rig.data.bones if b.use_deform
              and not b.name.startswith(("Knee", "Elbow"))]

    def deltas():
        out = {}
        for f in probe:
            scene.frame_set(f)
            for n in deform:
                out[(n, f)] = local_delta(rig.pose.bones[n])
        return out

    before = deltas()
    wide_rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
    if bpy.ops.efb.switch_build(variant="biped_slim_arm", rig=rig.name) != {"FINISHED"}:
        fail("switch to slim failed")
    rig = bpy.data.objects[rig.name] if rig.name in bpy.data.objects else \
        next(o for o in bpy.data.objects if o.type == "ARMATURE")

    scene.collection.children.link(bpy.data.collections.new("fresh"))
    active = bpy.context.view_layer.objects.active
    bpy.ops.efb.generate_rig(variant="biped_slim_arm", body_mesh=False)
    fresh = bpy.context.view_layer.objects.active
    worst = max(max(abs(a - b) for ra, rb in zip(rig.data.bones[n].matrix_local,
                                                   fresh.data.bones[n].matrix_local)
                    for a, b in zip(ra, rb)) for n in fresh.data.bones.keys())
    seat = 0.0
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type == "CHILD_OF":
                want = fresh.pose.bones[pb.name].constraints[c.name].inverse_matrix
                seat = max(seat, max(abs(a - b) for ra, rb in zip(c.inverse_matrix, want)
                                     for a, b in zip(ra, rb)))
    bpy.data.objects.remove(fresh)
    bpy.context.view_layer.objects.active = rig = active
    print("switch: rests vs fresh slim %.2e, Child Of inverses %.2e" % (worst, seat))
    if worst > 1e-5 or seat > 1e-5:
        fail("the switched rig is not the slim rig")

    now = {(fc.data_path, fc.array_index): [tuple(k.co) for k in fc.keyframe_points]
           for fc in action_curves(rig.animation_data.action, rig)
           if (fc.data_path, fc.array_index) in keys}
    if now != keys:
        fail("switching moved FK keys")
    after = deltas()
    drift = max(max(abs(a - b) for ra, rb in zip(before[k], after[k]) for a, b in zip(ra, rb))
                for k in before)
    bodies = bodies_of(rig)
    print("switch: pose deltas moved %.2e, bodies %s" % (drift, [
        (o.name, o.get("efb_body"), len(o.data.vertices)) for o in bodies]))
    if drift > 1e-5:
        fail("switching changed the pose deltas")
    if not bodies or bodies[0].get("efb_body") != "biped_slim_arm":
        fail("the body was not swapped")

    table_frames = list(range(scene.frame_end + 1))

    class _T:
        frames = table_frames

    ik_flip_holds(rig, _T)

    bpy.ops.efb.switch_build(variant="biped", rig=rig.name)
    back = max(max(abs(a - b) for ra, rb in zip(rig.data.bones[n].matrix_local, wide_rest[n])
                   for a, b in zip(ra, rb)) for n in wide_rest)
    print("switch back: rests vs the original wide %.2e" % back)
    if back > 1e-5:
        fail("slim -> wide does not come back")


def export_matches(path, arm):
    """The add-on's own Epic Fight export of the imported action is the json the command
    line writes straight from the file: same joints, same keys, the seam markers the rig
    slid included."""
    import tempfile

    from efb import quat as Q
    from efb.animjson import load

    out = os.path.join(tempfile.mkdtemp(), "export.json")
    res = bpy.ops.efb.export_animation(filepath=out, transform_format="attributes",
                                       joints="ALL")
    if res != {"FINISHED"}:
        fail("export returned %s" % res)
    got = load(out)
    want, _ = retarget(load_motion(path, fps=30.0), arm,
                       RetargetOptions(markers=True, static_tracks=True))
    if sorted(got.names) != sorted(want.names):
        fail("export writes %s, the retarget %s" % (got.names, want.names))
    worst_rot = worst_loc = 0.0
    for track in got.tracks:
        ref = want.track(track.name)
        if len(ref) != len(track):
            fail("%s: %d keys exported, %d retargeted" % (track.name, len(track), len(ref)))
        for k, r in zip(track.keys, ref.keys):
            worst_rot = max(worst_rot, math.degrees(Q.angle(k.rot, r.rot)))
            worst_loc = max(worst_loc, max(abs(x - y) for x, y in zip(k.loc, r.loc)))
    print("export vs retarget json: rot %.4f deg, loc %.2e m" % (worst_rot, worst_loc))
    if worst_rot > 0.01 or worst_loc > 1e-4:
        fail("the exported clip is not the retargeted one")


def render(rig, out_dir, frames):
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = 420
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("w") if scene.world is None else scene.world
    scene.world = world
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = 35
    bpy.ops.mesh.primitive_plane_add(size=6.0)
    for f in frames:
        scene.frame_set(f)
        hips = rig.matrix_world @ rig.pose.bones["Root"].head
        target = hips.copy()
        target.z = 0.9
        from mathutils import Vector
        cam.location = target + Vector((2.6, -3.6, 0.6))
        direction = target - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(out_dir, "frame_%04d.png" % f)
        bpy.ops.render.render(write_still=True)
    print("rendered", len(frames), "frames to", out_dir)


main()
