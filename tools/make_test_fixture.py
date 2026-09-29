"""Rebuild tests/data/squat_hpe_results.pt.

A GEM-X-shaped result (torch.save of {"body_params_global": {global_orient, body_pose,
transl, ...}, "K_fullimg", "net_outputs"}) made from real motion that may be redistributed:
SOMA-X's assets/example_animation.npy, Apache-2.0. The conversion follows SOMA-X's own
tests/test_pose_inversion.py::_load_motion step for step - remap its 94 joints onto the 78,
world = FK(local) @ t_pose_world^T, back to local, global_orient = local[:, 1],
body_pose = local[:, 2:], transl = the Hips translation - so the fixture holds exactly
what soma.pose() would be handed.

    python tools/make_test_fixture.py path/to/SOMA-X

Needs torch, numpy and scipy, and the SOMA-X checkout's git-lfs assets.
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "tests", "data", "squat_hpe_results.pt")
TABLE = os.path.join(HERE, os.pardir, "ef_blender", "efb", "data", "soma77.json")
FRAMES = slice(600, 660)


def joint_names(soma_x):
    """The 94 -> 77 name lists, read out of SOMA-X's test rather than copied here."""
    src = open(os.path.join(soma_x, "tests", "test_pose_inversion.py")).read()
    block = src[src.index("_NVSKEL93_NAMES = ["):src.index("# fmt: on")]
    ns = {}
    exec(block, ns)  # two list literals
    return ns["_NVSKEL93_NAMES"], ns["_NVSKEL77_NAMES"]


def main(soma_x):
    n93, n77 = joint_names(soma_x)
    idx = [0] + [n93.index(n) + 1 for n in n77]
    with open(TABLE) as fh:
        parents = json.load(fh)["parents"]
    rig = np.load(os.path.join(soma_x, "assets", "SOMA_neutral.npz"))
    orient = rig["t_pose_world__v0.2.0"].astype(np.float64)
    motion = np.load(os.path.join(soma_x, "assets", "example_animation.npy"))[FRAMES]
    motion = motion.astype(np.float64)
    local = motion[..., :3, :3][:, idx]
    transl = motion[:, 1, :3, 3]

    world = np.zeros_like(local)
    for j, p in enumerate(parents):
        world[:, j] = local[:, j] if p < 0 else world[:, p] @ local[:, j]
    world = world @ np.transpose(orient, (0, 2, 1))[None]
    for j, p in enumerate(parents):
        local[:, j] = world[:, j] if p < 0 else np.transpose(world[:, p], (0, 2, 1)) @ world[:, j]

    n = local.shape[0]
    rotvec = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(n, 78, 3)
    f32 = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    torch.save({
        "body_params_global": {
            "global_orient": f32(rotvec[:, 1]),
            "body_pose": f32(rotvec[:, 2:].reshape(n, 228)),
            "transl": f32(transl),
            "identity_coeffs": torch.zeros(n, 45),
            "scale_params": torch.zeros(n, 60),
        },
        "K_fullimg": torch.eye(3).repeat(n, 1, 1),
        "net_outputs": {
            "static_conf_logits": torch.zeros(1, n, 6),
            "note": "fixture: SOMA-X assets/example_animation.npy frames %d-%d "
                    "(Apache-2.0)" % (FRAMES.start, FRAMES.stop - 1),
        },
    }, OUT)
    print("wrote %s: %d frames" % (os.path.normpath(OUT), n))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
