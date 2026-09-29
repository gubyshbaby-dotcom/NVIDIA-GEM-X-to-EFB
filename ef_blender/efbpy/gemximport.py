"""File > Import > NVIDIA GEM-X Motion: a video's mocap onto the Epic Fight rig.

GEM-X leaves <output>/<video>/preprocess/hpe_results.pt behind; this reads it without
torch, retargets it in efb.gemx and hands the result to the same import every Epic Fight
clip goes through (efbpy.animops._do_import). So everything that path does, this does:
into an empty scene it spawns the rig and its body, onto a rig that is there it parks
whatever was playing on a muted NLA track, the clip lands on the FK controls, and the IK
handles are baked off it so any limb can be flipped to IK and worked on.

The keys are an Epic Fight clip from the start, so File > Export > Epic Fight Animation
writes it for the game with nothing further to do.

One thing a video does that a shipped clip does not: go past the rig's joint stops. The
Epic Fight preset is exactly the envelope of the mod's own clips, and a person folds an
elbow past 140 degrees or turns their head further than any of them. The game has no
stops, and the exporter reads the posed deform bones, so a stop left on would quietly
rewrite the capture. The import therefore measures, on the live rig, whether any stop
holds a bone off the clip, and if one does it switches the rig's Joint Limits off and says
so - the same stance the Epic Fight import takes with Replay. "Keep joint limits" clamps on
purpose instead. The IK bake runs after that decision, so the handles sit on the pose the
rig actually plays.
"""

import json
import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from efb import bundled
from efb.clip import pose_table_from_clip, read_document
from efb.gemx import ROOT_FULL, ROOT_IN_PLACE, RetargetOptions, check_armature, retarget
from efb.rig import COORD_BONE, Rig
from efb.soma import DEFAULT_FPS, SomaError, load_motion

from .animops import _armature, _rig_armature, _spawn_entity, _do_import, existing_rig
from .ops import BUILDS

__all__ = ["EFB_OT_import_gemx", "register", "unregister", "GEMX_PROP", "clip_label",
           "clamped_bones"]

GEMX_PROP = "efb_gemx"

CLAMP_TOLERANCE = 1e-4

MAX_CHECKED_FRAMES = 2000

_ROOT_MOTION = [
    (ROOT_FULL, "Full path", "The Root walks where the person walked, scaled to the "
                             "biped's legs"),
    (ROOT_IN_PLACE, "In place", "Only the Root's height moves. What looping locomotion "
                                "in Epic Fight usually wants"),
]


def clip_label(path) -> str:
    """The name an action gets. GEM-X calls every result hpe_results.pt and files it in
    outputs/demo_soma/<video>/ (preprocess/ in older layouts), so the video's name is the
    one worth keeping."""
    path = os.path.normpath(os.path.abspath(str(path)))
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem != "hpe_results":
        return stem
    folder = os.path.dirname(path)
    if os.path.basename(folder) == "preprocess":
        folder = os.path.dirname(folder)
    return os.path.basename(folder) or stem


class _ImportProxy:
    """What animops._do_import reads off an operator. Its `fps` has to be the clip's
    resolved rate, which is not a number the user typed, so it is handed over here
    rather than written back into the operator's redo panel."""

    def __init__(self, op, fps):
        self._op = op
        self.fps = fps
        for name in ("variant", "body_mesh", "bake_ik", "entity", "jar", "armature_file"):
            setattr(self, name, getattr(op, name))

    def report(self, kind, message):
        self._op.report(kind, message)


class EFB_OT_import_gemx(bpy.types.Operator, ImportHelper):
    """Retarget an NVIDIA GEM-X video mocap (hpe_results.pt), or any SOMA motion (.npz,
    .bvh), onto the Epic Fight rig"""
    bl_idname = "efb.import_gemx"
    bl_label = "Import GEM-X Motion"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".pt"
    filter_glob: StringProperty(default="*.pt;*.pth;*.npz;*.bvh", options={"HIDDEN"})

    source_fps: FloatProperty(
        name="Video rate", default=DEFAULT_FPS, min=1.0, max=1000.0,
        description="Frame rate of the original video you gave GEM-X - not of the 30 fps "
                    "copy its demo writes next to the results, which keeps every frame. "
                    "hpe_results.pt does not record it; a BVH carries its own")
    fps: FloatProperty(
        name="Clip rate", default=0.0, min=0.0, max=240.0,
        description="Rate of the clip written. 0 keeps the video's, up to 60")
    frame_start: IntProperty(name="First frame", default=0, min=0,
                             description="First video frame to use")
    frame_end: IntProperty(name="Last frame", default=-1, min=-1,
                           description="Last video frame to use, -1 for the end")
    smoothing: FloatProperty(
        name="Smoothing", default=1.0, min=0.0, max=10.0,
        description="Gaussian filter width in video frames. 0 keeps GEM-X's output as it "
                    "is")

    root_motion: EnumProperty(name="Root motion", items=_ROOT_MOTION, default=ROOT_FULL)
    face_forward: BoolProperty(
        name="Face forward", default=True,
        description="Turn the clip so the first frame faces the rig's front, whatever "
                    "way the camera saw it")
    start_at_origin: BoolProperty(
        name="Start at origin", default=True,
        description="Slide the clip so it starts on the rig's origin")
    ground: BoolProperty(
        name="Feet on the floor", default=True,
        description="Drop the clip so the biped's own feet touch the floor")
    subject_height: FloatProperty(
        name="Person's height", default=0.0, min=0.0, max=3.0, unit="LENGTH",
        description="Height of the filmed person. 0 estimates it from the clip. Sets "
                    "how far a stride carries the biped")

    hinge: BoolProperty(
        name="Hinge elbows and knees", default=True,
        description="Fold elbows and knees about their one hinge axis, as the biped's "
                    "boxes and the rig's pin expect. Off copies the forearm's roll too, "
                    "and the import raises Replay to keep it")
    tools: BoolProperty(
        name="Wrist to Tool", default=True,
        description="Turn Tool_R / Tool_L with the real wrist, so a held item follows "
                    "the hand")
    clavicle: FloatProperty(
        name="Collarbones", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
        description="How much of the collarbones' motion the shoulders take")
    keep_limits: BoolProperty(
        name="Keep joint limits", default=False,
        description="Let the rig's joint stops clamp the capture where it goes past "
                    "them - and the export with it. Off, the import switches the rig's "
                    "Joint Limits off when they would change the clip, and says so")

    variant: EnumProperty(name="Build", items=BUILDS, default=bundled.DEFAULT_VARIANT,
                          description="Which build to spawn when the scene has no rig "
                                      "yet. Ignored when a rig is already there")
    body_mesh: BoolProperty(name="Body Mesh", default=True,
                            description="Build the skinned body with a spawned rig")
    bake_ik: BoolProperty(name="Bake IK", default=True,
                          description="Put every limb's IK handle on the result at each "
                                      "frame, so a limb can be switched to IK anywhere")
    entity: StringProperty(name="Entity", default="biped")
    jar: StringProperty(name="Armature from jar", default="", subtype="FILE_PATH")
    armature_file: StringProperty(name="Armature from json", default="",
                                  subtype="FILE_PATH")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        for title, props, closed in (
                ("Source", ("source_fps", "fps", "frame_start", "frame_end",
                            "smoothing"), False),
                ("Placement", ("root_motion", "face_forward", "start_at_origin", "ground",
                               "subject_height"), False),
                ("Body", ("hinge", "tools", "clavicle", "keep_limits"), False),
                ("Rig", ("variant", "body_mesh", "bake_ik"), False),
                ("Armature source (optional)", ("entity", "armature_file", "jar"), True)):
            panel = getattr(layout, "panel", None)
            if panel is None:
                body = layout.box()
                body.label(text=title)
            else:
                header, body = panel("efb_gemx_" + title.split()[0].lower(),
                                     default_closed=closed)
                header.label(text=title)
            if body is None:
                continue
            body.use_property_split = True
            for prop in props:
                body.prop(self, prop)

    def retarget_options(self) -> RetargetOptions:
        return RetargetOptions(
            fps=self.fps, start=self.frame_start, end=self.frame_end,
            smoothing=self.smoothing, root_motion=self.root_motion,
            face_forward=self.face_forward, start_at_origin=self.start_at_origin,
            ground=self.ground, subject_height=self.subject_height, hinge=self.hinge,
            tools=self.tools, clavicle=self.clavicle)

    def execute(self, context):
        try:
            obj = existing_rig(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        try:
            if obj is not None:
                armature = _rig_armature(obj, self.jar, self.entity, self.armature_file)
            else:
                armature = _armature(self.jar, _spawn_entity(self), self.armature_file)
            check_armature(armature)
            motion = load_motion(self.filepath, fps=self._file_fps())
            doc, info = retarget(motion, armature, self.retarget_options())
        except (SomaError, OSError, ValueError) as exc:
            self.report({"ERROR"}, "%s: %s" % (os.path.basename(self.filepath), exc))
            return {"CANCELLED"}

        proxy = _ImportProxy(self, info.fps_out)
        proxy.bake_ik = False
        result = _do_import(proxy, context, doc, clip_label(self.filepath), armature)
        if result != {"FINISHED"}:
            return result
        rig = context.view_layer.objects.active
        limits_off = self._settle_limits(context, rig, armature, doc, info.fps_out)
        baked = self._bake(context, rig, info.frames_out) if self.bake_ik else 0
        action = rig.animation_data.action if rig and rig.animation_data else None
        if action is not None:
            action[GEMX_PROP] = json.dumps({
                "file": self.filepath, "kind": info.kind, "frames": info.frames_out,
                "fps": info.fps_out, "subject_height": round(info.subject_height, 4),
                "estimated": info.height_estimated, "leg_scale": round(info.leg_scale, 5),
                "options": vars(self.retarget_options()), "notes": info.notes,
                "limits_switched_off": limits_off})
        extra = list(info.notes[-2:])
        if baked:
            extra.append("IK baked on %d limbs" % baked)
        if limits_off:
            extra.append("Joint Limits switched off - the capture goes past them on %s"
                         % ", ".join("%s (%d frames)" % (n, c)
                                     for n, c in sorted(limits_off.items())[:4]))
        self.report({"WARNING"} if limits_off else {"INFO"},
                    "GEM-X: %s%s" % (info.summary(), "; " + "; ".join(extra) if extra else ""))
        return {"FINISHED"}

    def _settle_limits(self, context, rig, armature, doc, fps) -> dict:
        """Switch the rig's stops off when they would change the clip. Returns the bones
        they held, with a frame count, or {} when nothing was touched."""
        from .rigedit import settings

        s = settings(rig)
        if self.keep_limits or s is None or not s.limits:
            return {}
        held = clamped_bones(context, rig, armature, doc, fps)
        if held:
            s.limits = False
            rig.update_tag()
            context.view_layer.update()
        return held

    def _bake(self, context, rig, frames) -> int:
        """What import_document would have done with bake_ik on, after the limits are
        settled."""
        from .ikbake import bake_frames, bake_ik

        done = bake_ik(rig, bake_frames(range(frames), 0, max(1, frames - 1)),
                       context=context)
        return len(done)

    def _file_fps(self):
        """A BVH knows its own rate; a GEM-X result does not, so the user's wins."""
        return None if self.filepath.lower().endswith(".bvh") else self.source_fps


def clamped_bones(context, rig, armature, doc, fps) -> dict:
    """{deform bone: frames it is held off the clip}, measured on the evaluated rig.

    The seam markers and the Tool sockets are left out: the rig slides the markers itself,
    and the sockets ride a holder that is half a millimetre loose whatever the clip.
    """
    model = Rig(armature, coord=COORD_BONE in rig.pose.bones)
    table = pose_table_from_clip(model, read_document(model, doc, fps))
    names = [n for n in model.deform_bones()
             if n in rig.pose.bones and not n.startswith(("Knee", "Elbow", "Tool"))]
    step = max(1, len(table.frames) // MAX_CHECKED_FRAMES)
    scene = context.scene
    saved = scene.frame_current
    held = {}
    for i in range(0, len(table.frames), step):
        scene.frame_set(int(table.frames[i]))
        for name in names:
            got, want = rig.pose.bones[name].matrix, table.world[name][i]
            if any(abs(got[r][c] - want[r, c]) > CLAMP_TOLERANCE
                   for r in range(3) for c in range(4)):
                held[name] = held.get(name, 0) + step
    scene.frame_set(saved)
    return held


CLASSES = (EFB_OT_import_gemx,)


def _import_menu(self, context):
    self.layout.operator(EFB_OT_import_gemx.bl_idname,
                         text="NVIDIA GEM-X Motion to Epic Fight (.pt, .npz, .bvh)")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_import_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
