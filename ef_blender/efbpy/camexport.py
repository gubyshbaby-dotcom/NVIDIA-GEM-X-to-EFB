"""Blender cameras -> VIX camera animations plus a shot manifest.

Scene contract, all plain custom properties so a shot is authored without any UI:

  scene["ef_origin"]          name of the player/origin object; the export is taken
                              relative to it. Its local +Y is the direction the rig
                              faces, its location is the entity's feet.
  scene["ef_origin_mode"]     "entity" (default) follows the player every frame;
                              "locked" pins to the first frame, as SetAnim's
                              lockOrgPos does; "locked_dynamic_y" pins the ground plane
                              but lets the height follow, as dynamicY does.
  camera["ef_start"]          first frame this camera owns (defaults to scene.frame_start)
  camera["ef_end"]            last frame it owns (defaults to scene.frame_end)
  camera["ef_priority"]       int, higher wins an overlap (default 0)
  camera["ef_name"]           output file stem (defaults to the object name)

With no camera carrying ef_start, camera-bound timeline markers are used instead: each
marker owns the stretch up to the next one. Ranges are frame-inclusive at both ends, so
back-to-back cameras share their boundary frame and leave no gap.

VIX locks the player's yaw once, at SetAnim, so the whole shot uses one rig frame taken
at the first frame. The origin's translation stays live per frame, matching
calculateFinalCameraPosition adding entity.getPosition every tick.
"""

import os

import bpy

from efb.shotplan import CameraRange, Shot, dumps_manifest
from efb.vixcam import dumps as dumps_camera
from efb.vixexport import (CameraKey, DEFAULT_DIGITS, DEFAULT_TOLERANCE, RigFrame,
                           build_animation, vertical_fov_degrees)

from .animscene import from_matrix as to_mat4

SCENE_ORIGIN = "ef_origin"
SCENE_MODE = "ef_origin_mode"
CAM_START, CAM_END = "ef_start", "ef_end"
CAM_PRIORITY, CAM_NAME = "ef_priority", "ef_name"

ORIGIN_MODES = ("entity", "locked", "locked_dynamic_y")
TILT_TOLERANCE = 0.05


def origin_at(mode, live, frozen):
    """Which origin the offset is taken from. Blender Z is Minecraft Y, so the
    dynamic-height mode keeps the live Z and freezes the ground plane."""
    if mode == "entity":
        return live
    if mode == "locked":
        return frozen
    return (frozen[0], frozen[1], live[2])


def scene_fps(scene) -> float:
    return scene.render.fps / scene.render.fps_base


def find_origin(scene):
    named = scene.get(SCENE_ORIGIN)
    if named:
        obj = scene.objects.get(named)
        if obj is None:
            raise ValueError("scene['%s'] names %r, which is not in the scene"
                             % (SCENE_ORIGIN, named))
        return obj
    rigs = [o for o in scene.objects if o.type == "ARMATURE"]
    if len(rigs) == 1:
        return rigs[0]
    for guess in ("Player", "Origin", "Rig"):
        if guess in scene.objects:
            return scene.objects[guess]
    raise ValueError("set scene['%s'] to the player object" % SCENE_ORIGIN)


def _unique(name, taken):
    if name not in taken:
        return name
    i = 1
    while "%s.%d" % (name, i) in taken:
        i += 1
    return "%s.%d" % (name, i)


def camera_ranges(scene):
    """[(object, name, first_frame, last_frame, priority)], in start order."""
    cams = [o for o in scene.objects if o.type == "CAMERA"]
    tagged = [o for o in cams if CAM_START in o or CAM_END in o]
    out, taken = [], set()
    if tagged:
        for obj in tagged:
            name = _unique(str(obj.get(CAM_NAME) or obj.name), taken)
            taken.add(name)
            out.append((obj, name,
                        int(obj.get(CAM_START, scene.frame_start)),
                        int(obj.get(CAM_END, scene.frame_end)),
                        int(obj.get(CAM_PRIORITY, 0))))
    else:
        marks = sorted((m for m in scene.timeline_markers if m.camera), key=lambda m: m.frame)
        for i, m in enumerate(marks):
            end = marks[i + 1].frame if i + 1 < len(marks) else scene.frame_end
            name = _unique(m.camera.name, taken)
            taken.add(name)
            out.append((m.camera, name, m.frame, end, 0))
    if not out:
        if len(cams) == 1:
            return [(cams[0], cams[0].name, scene.frame_start, scene.frame_end, 0)]
        raise ValueError("no camera carries '%s' and no camera-bound markers exist"
                         % CAM_START)
    for obj, name, a, b, _ in out:
        if b <= a:
            raise ValueError("camera %r has an empty frame range %d..%d" % (name, a, b))
    return sorted(out, key=lambda r: (r[2], r[1]))


def camera_fov(scene, cam_data) -> float:
    if cam_data.type != "PERSP":
        raise ValueError("camera %r is %s; VIX only has a perspective fov"
                         % (cam_data.name, cam_data.type))
    r = scene.render
    return vertical_fov_degrees(cam_data.lens, cam_data.sensor_width, cam_data.sensor_height,
                                cam_data.sensor_fit, r.resolution_x, r.resolution_y,
                                r.pixel_aspect_x, r.pixel_aspect_y)


def sample_camera(scene, cam_obj, origin_obj, frames, rig, fps, first_frame,
                  mode="entity", frozen=None):
    """One CameraKey per frame, times zeroed at frames[0]."""
    keys = []
    for f in frames:
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        cam = cam_obj.evaluated_get(dg)
        world = to_mat4(cam.matrix_world)
        live = to_mat4(origin_obj.evaluated_get(dg).matrix_world).to_translation()
        keys.append(CameraKey(
            time=(f - first_frame) / fps,
            pos=rig.pos(world.to_translation(), origin_at(mode, live, frozen or live)),
            rot=rig.angles(world),
            fov=camera_fov(scene, cam.data),
        ))
    return keys


def build_shot(scene, name=None, strict=True, tolerance=DEFAULT_TOLERANCE,
               digits=DEFAULT_DIGITS):
    """Sample the scene. Returns (Shot, {camera name: VixCameraAnimation})."""
    origin_obj = find_origin(scene)
    ranges = camera_ranges(scene)
    fps = scene_fps(scene)
    first_frame = min(r[2] for r in ranges)
    restore = scene.frame_current

    try:
        scene.frame_set(first_frame)
        dg = bpy.context.evaluated_depsgraph_get()
        origin_world = to_mat4(origin_obj.evaluated_get(dg).matrix_world)
        rig = RigFrame.from_matrix(origin_world)
        warnings = []
        if rig.residual > TILT_TOLERANCE:
            msg = ("origin %r is tilted %.3f deg off upright; VIX locks a yaw only"
                   % (origin_obj.name, rig.residual))
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

        mode = str(scene.get(SCENE_MODE) or "entity")
        if mode not in ORIGIN_MODES:
            raise ValueError("scene['%s'] must be one of %r" % (SCENE_MODE, ORIGIN_MODES))
        frozen = origin_world.to_translation()
        anims = {}
        for obj, cam_name, start, end, _ in ranges:
            if obj.data.shift_x or obj.data.shift_y:
                warnings.append("camera %r has a lens shift; Minecraft renders centred"
                                % cam_name)
            keys = sample_camera(scene, obj, origin_obj, range(start, end + 1), rig, fps,
                                 start, mode, frozen)
            anims[cam_name] = build_animation(keys, tolerance, digits)
    finally:
        scene.frame_set(restore)

    shot = Shot(
        name=name or (scene.name if scene.name != "Scene" else "shot"),
        ranges=[CameraRange(cam_name, (start - first_frame) / fps,
                            (end - first_frame) / fps, prio)
                for _, cam_name, start, end, prio in ranges],
        origin_mode=mode,
        origin_object=origin_obj.name,
        fps=fps,
        warnings=warnings,
    )
    return shot, anims


def export_shot(scene, out_dir, name=None, **kw):
    """Write <out_dir>/<shot>.json and <out_dir>/<shot>/<camera>.json."""
    shot, anims = build_shot(scene, name, **kw)
    cam_dir = os.path.join(out_dir, shot.name)
    os.makedirs(cam_dir, exist_ok=True)
    files = {}
    for cam_name, anim in anims.items():
        files[cam_name] = "%s/%s.json" % (shot.name, cam_name)
        with open(os.path.join(cam_dir, cam_name + ".json"), "wb") as fh:
            fh.write(dumps_camera(anim))
    path = os.path.join(out_dir, shot.name + ".json")
    with open(path, "wb") as fh:
        fh.write(dumps_manifest(shot, files))
    return path, shot


class EFB_OT_export_vix_shot(bpy.types.Operator):
    bl_idname = "efb.export_vix_shot"
    bl_label = "Export VIX Camera Shot"
    bl_options = {"REGISTER"}

    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    shot_name: bpy.props.StringProperty(name="Shot")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            path, shot = export_shot(context.scene, self.directory,
                                     self.shot_name or None)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for msg in shot.manifest()["warnings"]:
            self.report({"WARNING"}, msg)
        self.report({"INFO"}, "wrote %s (%d cameras)" % (path, len(shot.ranges)))
        return {"FINISHED"}


def _export_menu(self, context):
    self.layout.operator(EFB_OT_export_vix_shot.bl_idname, text="VIX Camera Shot (.json)")


CLASSES = (EFB_OT_export_vix_shot,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
