"""Operators for the animation importer and exporter.

Self-contained: `import efbpy.animops; efbpy.animops.register()` is enough, and it adds
its own entries to File > Import and File > Export.

Importing into an empty scene spawns the rig first, so File > Import > Epic Fight
Animation is one step from nothing to a character playing the clip. Only when the scene
has none: a rig that is already there is imported onto, never duplicated, and whatever it
was playing is parked on a muted NLA track rather than overwritten.

The clip lands on FK, which is what the file stores, and the IK handles are then baked off
that result so a limb can be switched either way at any frame without the pose moving.
"""

from __future__ import annotations

import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from efb import assets, bundled
from efb import clip as efc
from efb.animjson import ATTRIBUTES, MATRIX, loads
from efb.armature import Armature
from efb.jarsource import DEFAULT_JAR, EfJar
from efb.rig import COORD_BONE, Rig
from efb.rigdef import RIG_ID

from .animexport import export_file
from .animimport import clip_name, import_document
from .animscene import ARMATURE_PROP, build_rig, find_rig
from .ikbake import baked_limbs
from .ops import BUILDS, build_default_rig

__all__ = ["register", "unregister", "CLASSES", "read_clip", "existing_rig",
           "stash_action"]

_SAMPLING = [
    (efc.AUTO, "Automatic", "Keep the keys when Epic Fight can replay them, bake when it "
                            "cannot. Measured on the clip, not guessed"),
    (efc.KEYED, "Keyed", "One key per bone per authored keyframe, which is what the "
                         "shipped animations use. Exact only for LINEAR curves"),
    (efc.BAKED, "Baked", "One shared frame ladder. Needed for bezier curves, "
                         "constraints, IK or drivers"),
]

_FORMAT = [
    ("SOURCE", "As imported", "Whatever the file this action came from used"),
    (MATRIX, "Matrix", "16 floats per key, the parent-relative pose"),
    (ATTRIBUTES, "Attributes", "loc/rot/sca per key, the local delta"),
]

_COVERAGE = [
    (efc.JOINTS_ALL, "Every joint",
     "All of the armature's joints, keyed or not. 351 of the 383 shipped clips do this, "
     "and a joint left out of the file is pinned to the bind pose in game"),
    (efc.JOINTS_SOURCE, "As imported",
     "The joints the file this action came from had, plus anything keyed since. For "
     "re-exporting a clip that has to diff clean against the original, including one "
     "that was missing joints"),
    (efc.JOINTS_KEYED, "Keyed joints only",
     "Only the joints you keyed, plus whatever sits between them and the root. The rest "
     "sit at the bind pose in game, so a limb you never touched will not move"),
]

_JAR_HELP = ("Optional source to read the armature from. Blank uses the armature "
             "bundled with the addon. Read only - the addon never writes into a jar")
_ARMATURE_FILE_HELP = ("Optional source: an armature json on disk, for an armature no "
                       "jar carries. Wins over the jar above")
_BUILD_HELP = ("Which build to spawn when the scene has no rig yet. A clip does not say "
               "wide or slim. Ignored when a rig is already there")
_BODY_HELP = ("Build the skinned body with the new rig. Only applies when a rig is "
              "spawned")
_BAKE_HELP = ("Put every limb's IK handle on the FK result at each frame, so a limb can be "
              "switched to IK anywhere without the pose moving. Off leaves the handles at "
              "rest, which is faster on a long clip and cannot be posed on IK")


def _armature(jar_path, entity, path=""):
    if path:
        return Armature.from_file(path)
    return assets.armature(entity, jar_path or None)


def _rig_armature(obj, jar_path, entity, path=""):
    """The armature json a rig object was built from, unless one is named outright."""
    if not path and obj.get(ARMATURE_PROP):
        entity = obj[ARMATURE_PROP]
    return _armature(jar_path, entity, path)




def is_efb_rig(obj) -> bool:
    """Whether an import may key this object. Either builder's rig qualifies; an armature
    the addon did not make does not, or an import would key bones it has no joint for."""
    return (obj is not None and obj.type == "ARMATURE"
            and (obj.data.get("rig_id") == RIG_ID or bool(obj.get(ARMATURE_PROP))))


def existing_rig(context) -> "bpy.types.Object | None":
    """The rig this import lands on, or None when the scene has none to land on.

    None is the spawn signal, so the only case left to refuse is ambiguity: several rigs
    and nothing active says which, where guessing would key the wrong character.
    """
    active = context.view_layer.objects.active
    if is_efb_rig(active):
        return active
    found = [o for o in context.scene.objects if is_efb_rig(o)]
    if len(found) == 1:
        return found[0]
    if found:
        raise RuntimeError("%d Epic Fight rigs in the scene - select the one to import "
                           "onto" % len(found))
    return None


def stash_action(obj):
    """Park what the rig is playing on a muted NLA track of its own. Returns it, or None.

    A second import must not cost the animator the first clip, and an action with no user
    is gone at the next save; the strip is a real user and the track is one click from
    playing again. Muted because an unmuted strip would blend into the clip arriving now.
    """
    ad = obj.animation_data
    action = ad.action if ad else None
    if action is None:
        return None
    start = int(action.frame_range[0]) if hasattr(action, "frame_range") else 0
    track = ad.nla_tracks.new()
    track.name = action.name
    strip = track.strips.new(action.name, start, action)
    strip.name = action.name
    track.mute = True
    ad.action = None
    return action


def spawn_rig(context, armature, entity, jar="", body_mesh=True, coord=False):
    """The rig an import builds when the scene has none.

    A bundled player build gets the full control rig and its body - the thing Generate
    makes, which is what "and the rig appears" means. Anything else has no biped joint set
    and no body mesh to bind, so it gets the plain deform rig from its own json.
    """
    if entity in bundled.VARIANTS:
        mesh = assets.body_mesh(entity, jar or None) if body_mesh else None
        obj, _ = build_default_rig(context, armature, mesh, entity,
                                   assets.skin_path(entity))
    else:
        obj = build_rig(Rig(armature, coord=coord))
    context.view_layer.objects.active = obj
    context.view_layer.update()
    return obj




def read_clip(path):
    """(document, "") or (None, why). Never raises: the animator picks the file in a
    browser, so it can be any json at all, or no json."""
    try:
        with open(path, "rb") as fh:
            return loads(fh.read()), ""
    except Exception as exc:
        return None, "%s is not an Epic Fight animation: %s" % (os.path.basename(path),
                                                                _why(exc))


def _why(exc) -> str:
    """A read failure in words. A bare KeyError stringifies to a quoted key, which says
    nothing in a status bar."""
    if isinstance(exc, KeyError):
        return "a track has no %s field" % exc
    return str(exc) or exc.__class__.__name__


def _name_list(names, limit=6) -> str:
    shown = ", ".join(names[:limit])
    return shown + (", ..." if len(names) > limit else "")


def _nothing_to_key(doc, missing, armature) -> str:
    if not doc.tracks:
        return 'not an Epic Fight animation: the file has no "animation" tracks'
    return ("no track names a %s joint, so there is nothing to key: %s"
            % (armature.name, _name_list(missing)))


class EFB_OT_build_rig(bpy.types.Operator):
    """Build a plain deform rig from an Epic Fight armature json"""
    bl_idname = "efb.build_rig"
    bl_label = "Build Epic Fight Rig"
    bl_options = {"REGISTER", "UNDO"}

    entity: StringProperty(name="Entity", default="biped")
    jar: StringProperty(name="Armature from jar", default="", subtype="FILE_PATH",
                        description=_JAR_HELP)
    armature_file: StringProperty(name="Armature from json", default="",
                                  subtype="FILE_PATH", description=_ARMATURE_FILE_HELP)
    coord: BoolProperty(name="Coord bone", default=True,
                        description="Add the root-motion bone above Root. 43 of the "
                                    "shipped clips animate it")

    def execute(self, context):
        arm = _armature(self.jar, self.entity, self.armature_file)
        obj = build_rig(Rig(arm, coord=self.coord))
        context.view_layer.objects.active = obj
        self.report({"INFO"}, "%s: %d bones" % (obj.name, len(obj.data.bones)))
        return {"FINISHED"}


class EFB_OT_import_animation(bpy.types.Operator, ImportHelper):
    """Load an Epic Fight animation json. Onto the rig in the scene, or onto one built for
    it when the scene has none"""
    bl_idname = "efb.import_animation"
    bl_label = "Import Epic Fight Animation"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    variant: EnumProperty(name="Build", items=BUILDS, default=bundled.DEFAULT_VARIANT,
                          description=_BUILD_HELP)
    body_mesh: BoolProperty(name="Body Mesh", default=True, description=_BODY_HELP)
    bake_ik: BoolProperty(name="Bake IK", default=True, description=_BAKE_HELP)
    entity: StringProperty(name="Entity", default="biped")
    jar: StringProperty(name="Armature from jar", default="", subtype="FILE_PATH",
                        description=_JAR_HELP)
    armature_file: StringProperty(name="Armature from json", default="",
                                  subtype="FILE_PATH", description=_ARMATURE_FILE_HELP)
    fps: FloatProperty(name="Rate", default=0.0, min=0.0,
                       description="0 picks the rate that puts every timestamp on a "
                                   "whole frame, which is 60 for the shipped clips")

    def draw(self, context):
        _draw_armature_source(self.layout, self, ("entity", "armature_file", "jar"),
                              ("fps", "variant", "body_mesh", "bake_ik"))

    def execute(self, context):
        doc, why = read_clip(self.filepath)
        if doc is None:
            self.report({"ERROR"}, why)
            return {"CANCELLED"}
        return _do_import(self, context, doc, self.filepath)


class EFB_OT_import_from_jar(bpy.types.Operator):
    """Load an Epic Fight animation straight out of the mod jar"""
    bl_idname = "efb.import_from_jar"
    bl_label = "Import Animation From Jar"
    bl_options = {"REGISTER", "UNDO"}

    clip: StringProperty(name="Clip", default="biped/living/walk.json")
    jar: StringProperty(name="Read from jar", default=DEFAULT_JAR, subtype="FILE_PATH",
                        description="The Epic Fight jar to read the clip out of. "
                                    "Opened read only")
    variant: EnumProperty(name="Build", items=BUILDS, default=bundled.DEFAULT_VARIANT,
                          description=_BUILD_HELP)
    body_mesh: BoolProperty(name="Body Mesh", default=True, description=_BODY_HELP)
    bake_ik: BoolProperty(name="Bake IK", default=True, description=_BAKE_HELP)
    fps: FloatProperty(name="Rate", default=0.0, min=0.0)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        with EfJar(self.jar or None) as jar:
            doc = jar.clip(self.clip)
            entity = jar.entity_for(self.clip)
            armature = None if entity == "biped" else jar.armature(entity)
        return _do_import(self, context, doc, self.clip, armature, entity)


def _spawn_entity(op, entity=None):
    """Which armature a spawned rig is built from. A named entity wins; otherwise the
    clip has not said wide or slim and the Build enum decides."""
    named = entity or (getattr(op, "entity", "") or "")
    if named and named != bundled.DEFAULT_VARIANT:
        return named
    return getattr(op, "variant", bundled.DEFAULT_VARIANT)


def _do_import(op, context, doc, source, armature=None, entity=None):
    """The whole import: find or build the rig, park what it was playing, key the clip.

    Everything that can refuse is settled before a single datablock is made, so a file
    that turns out not to be a clip for this armature leaves the scene exactly as it was.
    """
    try:
        obj = existing_rig(context)
    except RuntimeError as exc:
        op.report({"ERROR"}, str(exc))
        return {"CANCELLED"}

    spawn = _spawn_entity(op, entity)
    jar = getattr(op, "jar", "")
    if armature is None:
        armature = (_rig_armature(obj, jar, getattr(op, "entity", "biped"),
                                  getattr(op, "armature_file", ""))
                    if obj is not None else _armature(jar, spawn,
                                                      getattr(op, "armature_file", "")))
    has_coord = COORD_BONE in obj.pose.bones if obj is not None else \
        any(t.name == COORD_BONE for t in doc.tracks)
    rig = Rig(armature, coord=has_coord)
    missing = [t.name for t in doc.tracks if t.name not in rig]
    if len(missing) == len(doc.tracks):
        op.report({"ERROR"}, _nothing_to_key(doc, missing, armature))
        return {"CANCELLED"}

    if obj is None:
        try:
            obj = spawn_rig(context, armature, spawn, jar,
                            getattr(op, "body_mesh", True), has_coord)
        except Exception as exc:
            op.report({"ERROR"}, "cannot build a rig for this clip: %s" % exc)
            return {"CANCELLED"}
        built = True
    else:
        context.view_layer.objects.active = obj
        built = False

    rig = Rig(armature, coord=COORD_BONE in obj.pose.bones)
    missing = [t.name for t in doc.tracks if t.name not in rig]
    parked = stash_action(obj)
    action, clip = import_document(obj, rig, doc, fps=op.fps or None,
                                   name=clip_name(source), source=source,
                                   bake_ik=getattr(op, "bake_ik", True))
    keyed = sum(1 for f in clip.frames.values() if f)
    baked = baked_limbs(obj, action)
    op.report({"INFO"}, "%s on %s: %d bones keyed at %g fps%s%s%s%s"
              % (action.name, obj.name, keyed, clip.meta.fps,
                 ", rig built" if built else "",
                 ", IK baked on %d limbs" % len(baked) if baked else "",
                 ", %d track%s the rig has no bone for (%s)"
                 % (len(missing), "" if len(missing) == 1 else "s", _name_list(missing))
                 if missing else "",
                 ", %s parked on an NLA track" % parked.name if parked else ""))
    _report_shear(op, clip)
    return {"FINISHED"}


SHEAR_VISIBLE = efc.SHEAR_VISIBLE


def _report_shear(op, clip):
    """Say what a shear the pose bones cannot hold actually costs, in metres.

    Epic Fight stores a full 4x4 per joint and a pose bone decomposes to location,
    rotation and scale, so a sheared key lands on the nearest pose that is not sheared.
    """
    if not clip.unrepresentable:
        return
    worst = max(clip.shear_metres.values(), default=0.0)
    names = _name_list(sorted(clip.unrepresentable))
    op.report({"WARNING"} if worst > SHEAR_VISIBLE else {"INFO"},
              "%d joints carry a shear no pose bone can hold (%s); the imported pose "
              "sits up to %.3g mm off the file there"
              % (len(clip.unrepresentable), names, worst * 1000.0))


class EFB_OT_export_animation(bpy.types.Operator, ExportHelper):
    """Write the active action as an Epic Fight animation json"""
    bl_idname = "efb.export_animation"
    bl_label = "Export Epic Fight Animation"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    sampling: EnumProperty(name="Sampling", items=_SAMPLING, default=efc.AUTO)
    step: FloatProperty(name="Bake step", default=1.0, min=0.01,
                        description="Frames between baked samples")
    joints: EnumProperty(name="Joints", items=_COVERAGE, default=efc.JOINTS_ALL,
                         description="Which joints get a track in the file")
    transform_format: EnumProperty(name="Format", items=_FORMAT, default="SOURCE")
    entity: StringProperty(name="Entity", default="biped")
    jar: StringProperty(name="Armature from jar", default="", subtype="FILE_PATH",
                        description=_JAR_HELP)
    armature_file: StringProperty(name="Armature from json", default="",
                                  subtype="FILE_PATH", description=_ARMATURE_FILE_HELP)
    fps: FloatProperty(name="Rate", default=0.0, min=0.0,
                       description="0 uses the rate the action was imported at, or the "
                                   "scene rate")

    def draw(self, context):
        _draw_armature_source(self.layout, self,
                              ("entity", "armature_file", "jar"),
                              ("sampling", "step", "joints", "transform_format", "fps"))

    def execute(self, context):
        try:
            obj = find_rig(context.view_layer.objects.active)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        armature = _rig_armature(obj, self.jar, self.entity, self.armature_file)
        fmt = None if self.transform_format == "SOURCE" else self.transform_format
        _, info = export_file(self.filepath, obj, armature, mode=self.sampling,
                              step=self.step, format=fmt, fps=self.fps or None,
                              coverage=self.joints)
        self.report({"INFO"}, "wrote %s: %d tracks, %d keys, %s, drift %.4g blocks"
                    % (self.filepath, info["tracks"], info["keys"], info["mode"],
                       info.get("lerp_drift", info.get("drift", 0.0))))
        return {"FINISHED"}


def _draw_armature_source(layout, op, source_props, main_props):
    """The armature-source fields go in a closed sub-panel of their own.

    A file path sitting loose in a file browser reads as the destination, which is what
    the pre-filled jar field used to do in the export dialog. Nothing at the top level of
    these dialogs is a path any more.
    """
    layout.use_property_split = True
    layout.use_property_decorate = False
    for prop in main_props:
        layout.prop(op, prop)
    panel = getattr(layout, "panel", None)
    if panel is None:
        box = layout.box()
        box.label(text="Armature source (optional)")
        body = box
    else:
        header, body = panel("efb_armature_source", default_closed=True)
        header.label(text="Armature source (optional)")
    if body is None:
        return
    body.use_property_split = True
    for prop in source_props:
        body.prop(op, prop)


CLASSES = (EFB_OT_build_rig, EFB_OT_import_animation, EFB_OT_import_from_jar,
           EFB_OT_export_animation)


def _import_menu(self, context):
    self.layout.operator(EFB_OT_import_animation.bl_idname, text="Epic Fight Animation (.json)")


def _export_menu(self, context):
    self.layout.operator(EFB_OT_export_animation.bl_idname, text="Epic Fight Animation (.json)")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_import_menu)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
    bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
