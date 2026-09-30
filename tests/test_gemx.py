"""The GEM-X translator in plain python: no Blender, no torch, no numpy.

    python3 -m unittest discover -s tests -v

tests/data/squat_hpe_results.pt is a real torch.save file in GEM-X's layout, built by
tools/make_test_fixture.py from frames 600-659 of SOMA-X's assets/example_animation.npy
(Apache-2.0) through the same steps SOMA-X's own tests use to turn it into pose
parameters. tests/blender_gemx.py is the Blender half of the gate.
"""

import io
import math
import os
import pickle
import struct
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "ef_blender"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "tools"))

from efb import bundled  # noqa: E402
from efb import quat as Q  # noqa: E402
from efb.animjson import dumps, loads  # noqa: E402
from efb.clip import pose_table_from_clip, read_document  # noqa: E402
from efb.gemx import (KEYS_PER_BONE, KEYS_SHARED, LOC_PER_DEGREE,  # noqa: E402
                      ROOT_IN_PLACE, SOMA_TO_EF, RetargetOptions, check_armature,
                      retarget)
from efb.rig import Rig  # noqa: E402
from efb.soma import (SomaError, SomaSkeleton, load_motion,  # noqa: E402
                      load_with_camera, motion_from_bvh, motion_from_params)
from efb.tensorio import Opaque, read_torch  # noqa: E402

FIXTURE = os.path.join(HERE, "data", "squat_hpe_results.pt")
ARM = bundled.armature("biped")
SK = SomaSkeleton.reference()


def params(frames=3, overrides=None, go=(0.0, 0.0, 0.0), transl=(0.0, 0.95, 0.0)):
    """T-pose parameters, with {soma joint name: rotvec} laid over every frame."""
    bp = [[0.0] * 228 for _ in range(frames)]
    for name, rv in (overrides or {}).items():
        j = SK.index[name] - 2
        for f in range(frames):
            bp[f][3 * j:3 * j + 3] = list(rv)
    return ([list(go)] * frames, bp, [list(transl)] * frames)


def solve(opts=None, **kw):
    """Arms by their angles (hands=0) unless told otherwise: most of these tests are about
    the angle transfer and the hinge, which the hand placement would reshape."""
    go, bp, tr = params(**kw)
    motion = motion_from_params(go, bp, tr, fps=30.0)
    doc, rep = retarget(motion, ARM, opts or RetargetOptions(smoothing=0.0, hands=0.0))
    rig = Rig(ARM, coord=False)
    table = pose_table_from_clip(rig, read_document(rig, doc, rep.fps_out))
    return doc, rep, table


def y_axis(m):
    r = m.rows()
    return (r[0][1], r[1][1], r[2][1])


def close(a, b, tol=1e-3):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class TestSomaTable(unittest.TestCase):
    def test_tpose(self):
        self.assertEqual(len(SK), 78)
        self.assertEqual(SK.names[1], "Hips")
        self.assertTrue(close(SK.direction("RightArm", "RightForeArm"), (-1, 0, 0), 1e-4))
        self.assertTrue(close(SK.direction("LeftArm", "LeftForeArm"), (1, 0, 0), 1e-4))
        self.assertTrue(close(SK.direction("RightLeg", "RightShin"), (0, -1, 0), 1e-4))
        self.assertEqual(SK.tpose_orient[0], (1.0, 0.0, 0.0, 0.0))

    def test_soma_to_ef_axes(self):
        up, fwd, left = (Q.rotate(SOMA_TO_EF, v) for v in
                         ((0, 1, 0), (0, 0, 1), (1, 0, 0)))
        self.assertTrue(close(up, (0, 0, 1), 1e-9))
        self.assertTrue(close(fwd, (0, 1, 0), 1e-9))
        self.assertTrue(close(left, (-1, 0, 0), 1e-9))


class TestTensorIO(unittest.TestCase):
    def test_reads_gemx_file(self):
        tree = read_torch(FIXTURE)
        g = tree["body_params_global"]
        self.assertEqual(g["global_orient"].shape, (60, 3))
        self.assertEqual(g["body_pose"].shape, (60, 228))
        self.assertEqual(tree["K_fullimg"].shape, (60, 3, 3))
        self.assertIn("SOMA-X", tree["net_outputs"]["note"])

    def test_hostile_pickle_runs_nothing(self):
        marker = os.path.join(tempfile.mkdtemp(), "pwned")

        class Evil:
            def __reduce__(self):
                return (os.system, ("touch %s" % marker,))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"x": Evil()}, protocol=2))
        path = os.path.join(tempfile.mkdtemp(), "evil.pt")
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
        tree = read_torch(path)
        self.assertIsInstance(tree["x"], Opaque)
        self.assertFalse(os.path.exists(marker))

    def test_npz_params(self):
        def npy(shape, values):
            header = "{'descr': '<f4', 'fortran_order': False, 'shape': %r, }" % (shape,)
            header += " " * (63 - len(header) % 64) + "\n"
            return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header))
                    + header.encode("latin1") + struct.pack("<%df" % len(values), *values))

        path = os.path.join(tempfile.mkdtemp(), "motion.npz")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("global_orient.npy", npy((2, 3), [0.0] * 6))
            zf.writestr("body_pose.npy", npy((2, 228), [0.0] * 456))
            zf.writestr("transl.npy", npy((2, 3), [0.0, 0.95, 0.0] * 2))
            zf.writestr("fps.npy", npy((), [24.0]))
        motion = load_motion(path)
        self.assertEqual(motion.frame_count, 2)
        self.assertEqual(motion.fps, 24.0)

    def test_wrong_file_says_so(self):
        path = os.path.join(tempfile.mkdtemp(), "x.pt")
        with open(path, "wb") as fh:
            fh.write(b"not a zip")
        with self.assertRaises(SomaError):
            load_motion(path)


class TestRetarget(unittest.TestCase):
    def test_tpose_lands_in_a_t(self):
        _doc, _rep, t = solve()
        w = {n: t.world[n][0] for n in t.world}
        self.assertTrue(close(y_axis(w["Arm_R"]), (1, 0, 0)))
        self.assertTrue(close(y_axis(w["Hand_R"]), (1, 0, 0)))
        self.assertTrue(close(y_axis(w["Arm_L"]), (-1, 0, 0)))
        self.assertTrue(close(y_axis(w["Thigh_R"]), (0, 0, -1), 1e-2))
        self.assertTrue(close(y_axis(w["Torso"]), (0, 0, 1)))

    def test_elbow_folds_on_its_hinge(self):
        # SOMA's right forearm bent 90 degrees forward: -X swung onto +Z about +Y.
        doc, rep, t = solve(overrides={"RightForeArm": (0.0, math.pi / 2, 0.0)})
        self.assertTrue(close(y_axis(t.world["Hand_R"][0]), (0, 1, 0)))
        self.assertTrue(close(y_axis(t.world["Arm_R"][0]), (1, 0, 0)))
        rot = doc.track("Hand_R").keys[0].rot
        self.assertAlmostEqual(rot[2], 0.0, places=5)
        self.assertAlmostEqual(rot[3], 0.0, places=5)
        self.assertAlmostEqual(math.degrees(2 * math.atan2(rot[1], rot[0])), 96.31, places=1)
        self.assertLess(rep.hinge_error_degrees, 0.01)

    def test_knee_folds_backwards(self):
        # Right shin swung back 90 degrees: -Y onto -Z about +X.
        doc, _rep, t = solve(overrides={"RightShin": (math.pi / 2, 0.0, 0.0)})
        self.assertTrue(close(y_axis(t.world["Leg_R"][0]), (0, -1, 0), 2e-3))
        rot = doc.track("Leg_R").keys[0].rot
        self.assertLess(math.degrees(2 * math.atan2(rot[1], rot[0])), -89.0)

    def test_turned_to_face_forward(self):
        doc, rep, _t = solve(go=(0.0, math.pi / 2, 0.0))
        self.assertAlmostEqual(abs(rep.turned_degrees), 90.0, places=3)
        self.assertLess(math.degrees(Q.angle(doc.track("Root").keys[0].rot)), 1e-3)
        doc, rep, _t = solve(RetargetOptions(face_forward=False, smoothing=0.0),
                             go=(0.0, math.pi / 2, 0.0))
        self.assertAlmostEqual(math.degrees(Q.angle(doc.track("Root").keys[0].rot)),
                               90.0, places=3)

    def test_fixture_is_grounded_and_valid(self):
        motion = load_motion(FIXTURE, fps=30.0)
        doc, rep = retarget(motion, ARM)
        self.assertEqual(rep.frames_out, 60)
        self.assertTrue(rep.height_estimated)
        self.assertTrue(1.5 < rep.subject_height < 2.0)
        self.assertEqual(len(doc.tracks), 16)
        self.assertEqual(doc.tracks[0].name, "Root")
        rig = Rig(ARM, coord=False)
        table = pose_table_from_clip(rig, read_document(rig, doc, rep.fps_out))
        lowest = []
        for i in range(len(table.frames)):
            feet = []
            for leg in ("Leg_R", "Leg_L"):
                m = table.world[leg][i]
                tip = m.transform_point((0.0, 0.3864, 0.0))
                feet.append(min(tip[2], m.to_translation()[2]))
            lowest.append(min(feet))
        lowest.sort()
        self.assertAlmostEqual(lowest[int(round(0.02 * (len(lowest) - 1)))], 0.0, places=4)
        self.assertEqual(loads(dumps(doc)).to_json(), doc.to_json())
        for track in doc.tracks:
            self.assertTrue(all(b > a for a, b in zip(track.times, track.times[1:])))
            for k in track.keys:
                self.assertAlmostEqual(sum(v * v for v in k.rot), 1.0, places=4)

    def test_in_place_keeps_root_over_origin(self):
        motion = load_motion(FIXTURE, fps=30.0)
        doc, _rep = retarget(motion, ARM, RetargetOptions(root_motion=ROOT_IN_PLACE))
        rig = Rig(ARM, coord=False)
        table = pose_table_from_clip(rig, read_document(rig, doc, 30.0))
        heads = [m.to_translation() for m in table.world["Root"]]
        self.assertLess(max(abs(h[0]) for h in heads), 1e-4)
        self.assertLess(max(abs(h[1] - heads[0][1]) for h in heads), 1e-4)

    def test_game_json_has_every_joint(self):
        motion = load_motion(FIXTURE, fps=30.0)
        doc, _rep = retarget(motion, ARM, RetargetOptions(markers=True, static_tracks=True))
        self.assertEqual([t.name for t in doc.tracks], ARM.depth_first())
        knee = doc.track("Knee_R")
        self.assertLess(min(k.loc[2] for k in knee.keys), -0.05)

    def test_rate_and_window(self):
        motion = load_motion(FIXTURE, fps=30.0)
        _doc, rep = retarget(motion, ARM, RetargetOptions(fps=60.0))
        self.assertEqual(rep.frames_out, 119)
        _doc, rep = retarget(motion, ARM, RetargetOptions(start=10, end=19))
        self.assertEqual(rep.frames_out, 10)

    def test_height_override_sets_the_leg_scale(self):
        motion = load_motion(FIXTURE, fps=30.0)
        _doc, a = retarget(motion, ARM, RetargetOptions(subject_height=1.6))
        _doc, b = retarget(motion, ARM, RetargetOptions(subject_height=2.0))
        self.assertAlmostEqual(a.leg_scale / b.leg_scale, 2.0 / 1.6, places=6)

    def test_camera_space_clip_is_stood_up(self):
        motion = load_motion(FIXTURE, fps=30.0)
        upright, _ = retarget(motion, ARM)
        flip = Q.from_axis_angle((1.0, 0.0, 0.0), math.pi)
        motion.rotations = [[q if j == 0 or q is None else Q.mul(flip, q)
                             for j, q in enumerate(row)] for row in motion.rotations]
        motion.hips = [Q.rotate(flip, p) for p in motion.hips]
        doc, rep = retarget(motion, ARM)
        self.assertTrue(any("upside down" in n for n in rep.notes))
        for a, b in zip(upright.track("Chest").keys, doc.track("Chest").keys):
            self.assertLess(math.degrees(Q.angle(a.rot, b.rot)), 1e-3)

    def test_ungrounded_clip_is_not_measured(self):
        motion = load_motion(FIXTURE, fps=30.0)
        motion.hips = [(x, y + 3.0, z) for x, y, z in motion.hips]
        doc, rep = retarget(motion, ARM)
        self.assertFalse(rep.height_estimated)
        self.assertTrue(any("not grounded" in n for n in rep.notes))
        rig = Rig(ARM, coord=False)
        table = pose_table_from_clip(rig, read_document(rig, doc, rep.fps_out))
        lowest = min(table.world[leg][i].transform_point((0.0, 0.3864, 0.0))[2]
                     for i in range(len(table.frames)) for leg in ("Leg_R", "Leg_L"))
        self.assertAlmostEqual(lowest, 0.0, delta=0.02)

    def test_body_map_pulls_only_inside_the_shoulders(self):
        from efb.gemx import _Solver
        go, bp, tr = params()
        body = _Solver(motion_from_params(go, bp, tr, 30.0), ARM, RetargetOptions()).body
        self.assertAlmostEqual(body.biped, 0.375, places=3)
        self.assertAlmostEqual(body.pull(body.human), 0.0)
        self.assertAlmostEqual(body.pull(body.human + 0.2), 0.0)
        self.assertAlmostEqual(body.pull(0.0), body.excess)
        self.assertAlmostEqual(body.pull(-0.1), body.excess)
        self.assertGreater(body.excess, 0.15)

    def test_hands_keep_a_t_pose_straight(self):
        _doc, _rep, t = solve(RetargetOptions(smoothing=0.0, hands=1.0))
        for bone in ("Arm_R", "Hand_R"):
            got = y_axis(t.world[bone][0])
            self.assertLess(math.degrees(math.acos(min(1.0, got[0]))), 3.0)

    def test_hands_follow_the_body(self):
        """The person's right hand held across the chest, past the midline: copied angles
        leave the biped's on its own side of the body, hand placement carries it over."""
        pose = {"RightArm": (0.0, math.pi / 2, 0.0), "RightForeArm": (0.0, math.pi / 2, 0.0)}
        grip = {}
        for hands in (0.0, 1.0):
            _doc, _rep, t = solve(RetargetOptions(smoothing=0.0, hands=hands),
                                  overrides=pose)
            chest = t.world["Chest"][0].inverse()
            grip[hands] = chest.transform_point(t.world["Tool_R"][0].to_translation())
        # The biped's arm (0.57 m) is shorter than the span between its arm roots (0.75 m),
        # so it reaches over as far as it can rather than to the mapped point.
        self.assertGreater(grip[0.0][0], 0.05)
        self.assertLess(grip[1.0][0], -0.05)

    def test_reaching_arms_open_smoothly(self):
        from efb.gemx import EXTEND_BELOW, _extend
        t = math.radians(EXTEND_BELOW)
        self.assertEqual(_extend(t, 1.0), t)
        self.assertEqual(_extend(-2.0, 1.0), -2.0)
        self.assertAlmostEqual(math.degrees(_extend(math.radians(27.0), 1.0)), 13.77, places=2)
        self.assertEqual(_extend(math.radians(27.0), 0.0), math.radians(27.0))
        eps = 1e-6
        slope = (_extend(t - eps, 1.0) - _extend(t - 2 * eps, 1.0)) / eps
        self.assertAlmostEqual(slope, 1.0, places=4)
        self.assertLess(_extend(-0.3, 1.0), 0.0)

    def test_a_jab_opens_on_the_hinge(self):
        # Right forearm 27 degrees off straight, forward: the biped's opens to ~14.
        doc, _rep, t = solve(RetargetOptions(smoothing=0.0, hands=0.0),
                             overrides={"RightForeArm": (0.0, math.radians(27.0), 0.0)})
        rot = doc.track("Hand_R").keys[0].rot
        self.assertAlmostEqual(rot[2], 0.0, places=5)
        fold = math.degrees(2 * math.atan2(rot[1], rot[0]))
        self.assertAlmostEqual(fold, 13.77 + 6.31, delta=0.1)
        doc, _rep, t = solve(RetargetOptions(smoothing=0.0, hands=0.0, extend=0.0),
                             overrides={"RightForeArm": (0.0, math.radians(27.0), 0.0)})
        rot = doc.track("Hand_R").keys[0].rot
        self.assertAlmostEqual(math.degrees(2 * math.atan2(rot[1], rot[0])), 27.0 + 6.31,
                               delta=0.1)

    def test_not_a_biped(self):
        arm = bundled.armature("biped")
        arm.joints.pop("Tool_R")
        with self.assertRaises(SomaError):
            check_armature(arm)


class TestVideoCamera(unittest.TestCase):
    def test_camera_is_recovered_from_the_two_copies(self):
        """A camera 4 m in front of the person, 1.5 m up, looking at their hips: the
        camera-space copy of the body is built from it, and the recovered Blender camera
        has to look at the biped's pelvis from the same distance, scaled to the rig."""
        world = load_motion(FIXTURE, fps=30.0)
        cam = (0.3, 1.5, 4.0)
        look = Q.unit3(Q.sub3(world.hips[0], cam))
        # world -> camera (OpenCV: x right, y down, z forward), looking along `look`
        z = look
        x = Q.unit3(Q.cross3((0.0, -1.0, 0.0), z))
        y = Q.cross3(z, x)
        r = Q.from_rows((x, y, z))
        body = [[_rotvec(Q.mul(Q.conj(g[SK.parents[j]]), g[j])) for j in range(2, 78)]
                for g in _full(world)]
        inc = motion_from_params([_rotvec(Q.mul(r, g[1])) for g in _full(world)],
                                 [[v for rv in row for v in rv] for row in body],
                                 [Q.rotate(r, Q.sub3(h, cam)) for h in world.hips], 30.0)
        wld = motion_from_params([_rotvec(g[1]) for g in _full(world)],
                                 [[v for rv in row for v in rv] for row in body],
                                 world.hips, 30.0)
        _doc, rep = retarget(wld, ARM, RetargetOptions(smoothing=0.0), inc,
                             (1000.0, 1000.0, 640.0, 360.0))
        track = rep.camera
        self.assertEqual((track.width, track.height), (1280, 720))
        rig = Rig(ARM, coord=False)
        table = pose_table_from_clip(rig, read_document(rig, _doc, rep.fps_out))
        pelvis_h = Q.scale3(Q.add3(*[wld.positions(0, ("LeftLeg", "RightLeg"))[k]
                                     for k in ("LeftLeg", "RightLeg")]), 0.5)
        root = table.world["Root"][0].to_translation()
        to_root = Q.sub3(root, track.location[0])
        forward = Q.rotate(track.rotation[0], (0.0, 0.0, -1.0))
        self.assertAlmostEqual(Q.norm3(to_root), rep.leg_scale * Q.norm3(Q.sub3(pelvis_h, cam)),
                               places=3)
        hips_dir = Q.unit3(Q.sub3(Q.add3(root, Q.rotate(Q.mul(
            Q.from_axis_angle((0, 0, 1), math.radians(rep.turned_degrees)), SOMA_TO_EF),
            Q.scale3(Q.sub3(wld.hips[0], pelvis_h), rep.leg_scale))), track.location[0]))
        self.assertGreater(Q.dot3(forward, hips_dir), math.cos(math.radians(0.5)))


class TestReachingArms(unittest.TestCase):
    """A right arm filmed from its own side, the camera 4 m off along -X, its forearm
    tipped towards the lens: in the video it runs where the upper arm runs, in 3D it is
    bent. Rotations are about SOMA's Y (up) or Z; the T-pose arm points along -X."""

    CAM = (-4.0, 1.3, 0.2)

    def shoot(self, bp, incam=True, **opts):
        n = len(bp)
        world = motion_from_params([[0.0] * 3] * n, bp, [[0.0, 0.95, 0.0]] * n, fps=30.0)
        cam = None
        if incam:
            z = Q.unit3(Q.sub3(world.hips[0], self.CAM))
            x = Q.unit3(Q.cross3((0.0, -1.0, 0.0), z))
            r = Q.from_rows((x, Q.cross3(z, x), z))
            cam = motion_from_params([_rotvec(r)] * n, bp,
                                     [Q.rotate(r, Q.sub3(h, self.CAM)) for h in world.hips],
                                     30.0)
        return retarget(world, ARM, RetargetOptions(hands=0.0, **opts), cam,
                        (1000.0, 1000.0, 640.0, 360.0) if incam else None)

    def film(self, arm, fore, incam=True):
        _go, bp, _tr = params(frames=2, overrides={"RightArm": arm, "RightForeArm": fore})
        doc, rep = self.shoot(bp, incam)
        rot = doc.track("Hand_R").keys[0].rot
        return abs(math.degrees(2 * math.atan2(rot[1], rot[0]))) - 6.31, rep

    def jab(self, bend, incam=True):
        # upper arm straight forward (+Z), forearm turned `bend` degrees towards -X
        return self.film((0.0, math.pi / 2, 0.0), (0.0, -math.radians(bend), 0.0), incam)

    def test_a_jab_bent_in_depth_comes_out_straight(self):
        bend, rep = self.jab(50.0)
        self.assertLess(bend, 1.0)
        self.assertEqual(rep.arms_straightened, 2)

    def test_without_the_camera_only_the_curve_opens_it(self):
        bend, rep = self.jab(50.0, incam=False)
        self.assertAlmostEqual(bend, 40.12, delta=0.1)
        self.assertEqual(rep.arms_straightened, 0)

    def test_a_guard_keeps_its_fold(self):
        bend, _rep = self.jab(110.0)
        self.assertAlmostEqual(bend, 110.0, delta=0.1)

    def test_a_hanging_arm_is_not_reaching(self):
        # upper arm straight down, forearm tipped 50 degrees towards the lens
        bend, rep = self.film((0.0, 0.0, math.pi / 2), (0.0, 0.0, -math.radians(50.0)))
        self.assertAlmostEqual(bend, 50.0, delta=0.1)
        self.assertEqual(rep.arms_straightened, 0)

    def test_straightening_only_opens_the_hinge(self):
        """Frame by frame the camera opens the elbow and nothing else: the upper arm, its
        twist included, is the one it is with straightening off, and the forearm differs
        from it only about the hinge. Turning the forearm onto the straighter direction
        instead swung the limb's plane round as the arm straightened - near straight it is
        any plane - and the hinge spun the arm about itself with it, 55-77 degrees in a
        frame at a boxing clip's punches."""
        up, fore = SK.index["RightArm"] - 2, SK.index["RightForeArm"] - 2
        axis = Q.unit3((0.0, -1.0, 0.6))
        bp = []
        for f in range(14):
            row = [0.0] * 228
            row[3 * up:3 * up + 3] = [0.0, math.pi / 2, 0.0]
            row[3 * fore:3 * fore + 3] = list(Q.scale3(axis, math.radians(95.0 - 6.0 * f)))
            bp.append(row)
        cam, rep = self.shoot(bp)
        off, _ = self.shoot(bp, incam=False, extend=0.0)
        self.assertGreater(rep.arms_straightened, 5)
        opened = 0.0
        for a, b in zip(cam.track("Arm_R").keys, off.track("Arm_R").keys):
            self.assertLess(math.degrees(Q.angle(a.rot, b.rot)), 1e-4)
        for a, b in zip(cam.track("Hand_R").keys, off.track("Hand_R").keys):
            d = Q.mul(Q.conj(b.rot), a.rot)
            self.assertLess(max(abs(d[2]), abs(d[3])), 1e-6)
            opened = max(opened, math.degrees(Q.angle(a.rot, b.rot)))
        self.assertGreater(opened, 20.0)

    def test_no_camera_copy_no_camera(self):
        # the fixture holds body_params_global alone; the search must not hand it back
        # as the camera-space copy
        _world, incam, _k = load_with_camera(FIXTURE, 30.0)
        self.assertIsNone(incam)


def _full(motion):
    """Every joint's G, the ones the loader skipped filled from their parents."""
    out = []
    for row in motion.rotations:
        row = list(row)
        for j in range(len(row)):
            if row[j] is None:
                row[j] = row[SK.parents[j]]
        out.append(row)
    return out


def _rotvec(q):
    q = Q.normalize(q)
    if q[0] < 0.0:
        q = tuple(-v for v in q)
    s = math.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if s < 1e-12:
        return (0.0, 0.0, 0.0)
    ang = 2.0 * math.atan2(s, q[0])
    return tuple(v / s * ang for v in q[1:])


class TestKeyReduction(unittest.TestCase):
    """The json that goes straight to the game: Epic Fight lerps and nlerps between keys,
    so the check is Track.sample, the port of its own interpolation."""

    def check(self, mode, tol):
        motion = load_motion(FIXTURE, fps=30.0)
        dense, _ = retarget(motion, ARM, RetargetOptions(markers=True))
        thin, rep = retarget(motion, ARM, RetargetOptions(markers=True, keys=mode,
                                                          key_tolerance=tol))
        self.assertLess(rep.keys_kept, rep.keys_dense)
        worst_rot = worst_loc = 0.0
        for full in dense.tracks:
            kept = thin.track(full.name)
            self.assertEqual(kept.times[0], full.times[0])
            self.assertEqual(kept.times[-1], full.times[-1])
            for time, key in zip(full.times, full.keys):
                loc, rot, _scale = kept.sample(time).decompose()
                worst_rot = max(worst_rot, math.degrees(Q.angle(rot, key.rot)))
                worst_loc = max(worst_loc, max(abs(a - b) for a, b in zip(loc, key.loc)))
        self.assertLessEqual(worst_rot, tol + 1e-3)
        self.assertLessEqual(worst_loc, tol * LOC_PER_DEGREE)
        return thin, rep

    def test_per_bone_replays_inside_tolerance(self):
        thin, rep = self.check(KEYS_PER_BONE, 0.5)
        self.assertGreater(len({tuple(t.times) for t in thin.tracks}), 1)

    def test_shared_keys_line_up(self):
        thin, rep = self.check(KEYS_SHARED, 1.0)
        self.assertEqual(len({tuple(t.times) for t in thin.tracks}), 1)
        self.assertEqual(rep.key_frames, len(thin.tracks[0]))


class TestBvh(unittest.TestCase):
    def test_bvh_and_params_agree(self):
        """A BVH written in the exporters' convention (local = orient[p]^-1 G-step
        orient[j]) reads back as the same G the parameters give."""
        src = load_motion(FIXTURE, fps=30.0)
        path = os.path.join(tempfile.mkdtemp(), "clip.bvh")
        write_bvh(path, src, frames=range(0, 60, 6))
        back = motion_from_bvh(path)
        self.assertEqual(back.frame_count, 10)
        self.assertAlmostEqual(back.fps, 5.0, places=3)
        worst = 0.0
        for k, f in enumerate(range(0, 60, 6)):
            for name in ("Hips", "Chest", "Head", "RightForeArm", "LeftShin", "RightHand"):
                j = SK.index[name]
                worst = max(worst, Q.angle(src.rotations[f][j], back.rotations[k][j]))
            self.assertTrue(close(src.hips[f], back.hips[k], 1e-5))
        self.assertLess(math.degrees(worst), 1e-3)


def write_bvh(path, motion, frames):
    """Full 78-joint SOMA BVH, centimetres, ZYX channels."""
    sk = motion.skeleton
    children = {j: [c for c in range(len(sk)) if sk.parents[c] == j] for j in range(len(sk))}
    lines = ["HIERARCHY"]

    def joint(j, depth):
        pad = "  " * depth
        p = sk.parents[j]
        off = (0.0, 0.0, 0.0) if p < 0 else Q.rotate(Q.conj(sk.tpose_orient[p]),
                                                      sk.offsets[j])
        lines.append("%s%s %s" % (pad, "ROOT" if p < 0 else "JOINT", sk.names[j]))
        lines.append(pad + "{")
        lines.append("%s  OFFSET %.6f %.6f %.6f" % ((pad,) + tuple(v * 100 for v in off)))
        lines.append(pad + "  CHANNELS 6 Xposition Yposition Zposition Zrotation "
                           "Yrotation Xrotation")
        for c in children[j]:
            joint(c, depth + 1)
        if not children[j]:
            lines.extend([pad + "  End Site", pad + "  {", pad + "    OFFSET 0 0 0",
                          pad + "  }"])
        lines.append(pad + "}")

    joint(0, 0)
    frames = list(frames)
    lines += ["MOTION", "Frames: %d" % len(frames), "Frame Time: 0.2"]
    full = [j for j in range(len(sk))]
    for f in frames:
        g = list(motion.rotations[f])
        for j in full:
            if g[j] is None:
                g[j] = g[sk.parents[j]]
        world = [Q.mul(g[j], sk.tpose_orient[j]) for j in full]
        row = []
        for j in full:
            p = sk.parents[j]
            local = world[j] if p < 0 else Q.mul(Q.conj(world[p]), world[j])
            if j == SK.index["Hips"]:
                pos = motion.hips[f]
            elif p < 0:
                pos = (0.0, 0.0, 0.0)
            else:
                pos = Q.rotate(Q.conj(sk.tpose_orient[p]), sk.offsets[j])
            r = Q.to_rows(local)
            b = -math.asin(max(-1.0, min(1.0, r[2][0])))
            a = math.atan2(r[1][0], r[0][0])
            c = math.atan2(r[2][1], r[2][2])
            row += [v * 100 for v in pos] + [math.degrees(v) for v in (a, b, c)]
        lines.append(" ".join("%.9f" % v for v in row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


class TestCommandLine(unittest.TestCase):
    def test_names_the_json_after_the_video(self):
        import gemx_to_ef

        self.assertEqual(gemx_to_ef.default_output("/o/demo_soma/kick/hpe_results.pt"),
                         os.path.normpath("/o/demo_soma/kick/kick_epicfight.json"))
        self.assertEqual(
            gemx_to_ef.default_output("/o/kick/preprocess/hpe_results.pt"),
            os.path.normpath("/o/kick/kick_epicfight.json"))

    def test_writes_a_game_json(self):
        import gemx_to_ef

        out = os.path.join(tempfile.mkdtemp(), "squat.json")
        self.assertEqual(gemx_to_ef.main([FIXTURE, "-o", out, "--fps-in", "30"]), 0)
        with open(out, "rb") as fh:
            raw = fh.read()
        doc = loads(raw)
        self.assertEqual(doc.format, "attributes")
        self.assertEqual(len(doc.tracks), 20)
        self.assertIn(b"\r\n", raw)


if __name__ == "__main__":
    unittest.main()
