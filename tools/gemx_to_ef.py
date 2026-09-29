#!/usr/bin/env python3
"""NVIDIA GEM-X video mocap -> Epic Fight animation json, without Blender.

    python tools/gemx_to_ef.py outputs/demo_soma/myvideo/hpe_results.pt --fps-in 30

Plain python 3.8+, no torch, no numpy: it uses the add-on's own efb package, so the file
it writes is byte for byte what the Blender import followed by an ATTRIBUTES export would
carry for the same settings. Every joint gets a track, the Elbow_* / Knee_* seam markers
included, the way the shipped clips have them; drop the json into
assets/<modid>/animmodels/animations/... of a resource pack or mod and point an Epic Fight
animation at it.

Also reads an npz holding global_orient / body_pose / transl, and SOMA-skeleton BVH
(BONES-SEED, Kimodo, GEM-X's own export).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "ef_blender"))

from efb import bundled  # noqa: E402
from efb.animjson import save  # noqa: E402
from efb.armature import Armature  # noqa: E402
from efb.gemx import (KEYS_ALL, KEYS_PER_BONE, KEYS_SHARED, ROOT_FULL,  # noqa: E402
                      ROOT_IN_PLACE, RetargetOptions, retarget)
from efb.soma import SomaError, describe, load_motion  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="Retarget NVIDIA GEM-X (SOMA) motion onto the Epic Fight biped and "
                    "write an Epic Fight animation json.")
    p.add_argument("input", help="hpe_results.pt, an npz, or a SOMA .bvh")
    p.add_argument("-o", "--output", help="json to write (default: next to the input, "
                                          "named after the video)")
    p.add_argument("--fps-in", type=float, default=None,
                   help="frame rate of the original video given to GEM-X (hpe_results.pt "
                        "does not record it, and the demo's .mp4 copy is always 30; "
                        "default 30). Ignored for BVH")
    p.add_argument("--fps", type=float, default=0.0,
                   help="rate of the clip written; 0 keeps the source's, up to 60")
    p.add_argument("--start", type=int, default=0, help="first source frame")
    p.add_argument("--end", type=int, default=-1, help="last source frame, -1 = end")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="gaussian smoothing, sigma in source frames; 0 = off")
    p.add_argument("--in-place", action="store_true",
                   help="drop the horizontal path, keep only the Root's height")
    p.add_argument("--keep-facing", action="store_true",
                   help="do not turn the clip to face the rig's front")
    p.add_argument("--keep-origin", action="store_true",
                   help="do not slide the clip's start onto the origin")
    p.add_argument("--no-ground", action="store_true",
                   help="do not drop the clip onto the floor")
    p.add_argument("--height", type=float, default=0.0,
                   help="the filmed person's height in metres; 0 estimates it")
    p.add_argument("--no-hinge", action="store_true",
                   help="copy forearm and shin roll instead of folding on the hinge")
    p.add_argument("--no-tools", action="store_true",
                   help="leave Tool_R / Tool_L at rest in the hand")
    p.add_argument("--clavicle", type=float, default=1.0,
                   help="share of the collarbones' motion the shoulders take, 0..1")
    p.add_argument("--keys", choices=("per-bone", "shared", "all"), default="per-bone",
                   help="per-bone (default) keeps on each track only the keys Epic Fight's "
                        "own lerp cannot recreate; shared keeps the same frames on every "
                        "track; all keeps every frame")
    p.add_argument("--key-tolerance", type=float, default=0.5,
                   help="degrees a joint may stray between kept keys (default 0.5)")
    p.add_argument("--slim", action="store_true", help="the slim-armed (Alex) biped")
    p.add_argument("--armature", help="an Epic Fight armature json to retarget onto "
                                      "instead of the bundled biped")
    p.add_argument("--describe", action="store_true",
                   help="list what the input file holds and stop")
    return p.parse_args(argv)


def default_output(path):
    """<video>_epicfight.json next to GEM-X's own outputs for the video."""
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(os.path.abspath(path))
    if stem == "hpe_results":
        if os.path.basename(folder) == "preprocess":
            folder = os.path.dirname(folder)
        stem = os.path.basename(folder) or stem
    return os.path.join(folder, stem + "_epicfight.json")


def main(argv=None):
    a = parse(argv)
    if a.describe:
        print(describe(a.input))
        return 0
    armature = (Armature.from_file(a.armature) if a.armature
                else bundled.armature("biped_slim_arm" if a.slim else "biped"))
    opts = RetargetOptions(
        fps=a.fps, start=a.start, end=a.end, smoothing=a.smooth,
        root_motion=ROOT_IN_PLACE if a.in_place else ROOT_FULL,
        face_forward=not a.keep_facing, start_at_origin=not a.keep_origin,
        ground=not a.no_ground, subject_height=a.height, hinge=not a.no_hinge,
        tools=not a.no_tools, clavicle=a.clavicle, markers=True, static_tracks=True,
        keys={"per-bone": KEYS_PER_BONE, "shared": KEYS_SHARED, "all": KEYS_ALL}[a.keys],
        key_tolerance=a.key_tolerance)
    try:
        motion = load_motion(a.input, fps=a.fps_in)
        doc, info = retarget(motion, armature, opts)
    except (SomaError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    out = a.output or default_output(a.input)
    save(out, doc)
    print("wrote %s" % out)
    print("  %s" % info.summary())
    print("  %d tracks, turned %.1f deg to face forward, floor moved %.3f m"
          % (len(doc.tracks), info.turned_degrees, info.floor_offset))
    for note in info.notes:
        print("  note: %s" % note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
