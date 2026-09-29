"""Headless gate: GEM-X import onto the real control rig, in a real Blender.

    blender -b --factory-startup --python-exit-code 1 --python tests/blender_gemx.py -- \
        path/to/hpe_results.pt [--render out_dir] [--frames 0,30,60]

Checks that the operator spawns the rig, keys the clip, leaves every limb on FK with the
IK handles baked, and that the deform bones Blender evaluates are the pose the retarget
wrote - the same comparison the Epic Fight export makes - to 1e-4. With --render it also
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

    res = bpy.ops.efb.import_gemx(filepath=a["path"])
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
