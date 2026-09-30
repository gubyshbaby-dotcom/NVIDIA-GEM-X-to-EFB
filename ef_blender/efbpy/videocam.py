"""The phone's camera, standing where it stood, with the video behind it.

A retarget copies GEM-X's 3D pose, and the 3D pose is where a monocular estimate is
weakest: depth. GEM-X's keypoint overlay is 2D and always looks right; an arm GEM-X put
0.3 m too far forward looks the same in it, and only shows up when the rig is orbited.
So the import stands a camera exactly where the video's was - recovered from the two
copies of the body GEM-X saves, efb.gemx.CameraTrack - with the video as its background.
Through it the biped sits on the person frame by frame: where they part, the rig or the
estimate is wrong, and the other GEM-X renders (<video>_1_incam.mp4, _2_global.mp4) say
which.

The camera hangs off the rig, keyed in armature space at the rig's scale, so it frames the
biped as the clip's root placement leaves it and follows the rig object if that moves.
Background images show in the viewport's camera view, not in renders.
"""

import os

import bpy

from .animimport import _assign_slot, _curve

__all__ = ["build_camera", "find_video", "CAMERA_PROP", "camera_of"]

CAMERA_PROP = "efb_gemx_camera"

SENSOR = 36.0


def find_video(result_path, label):
    """GEM-X's copy of the input next to its result: <output>/<video>/<video>.mp4, or the
    older 0_input_video.mp4. None when neither is there."""
    folder = os.path.dirname(os.path.abspath(result_path))
    if os.path.basename(folder) == "preprocess":
        folder = os.path.dirname(folder)
    for name in (label + ".mp4", "0_input_video.mp4"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def camera_of(rig):
    return next((o for o in bpy.data.objects
                 if o.type == "CAMERA" and o.get(CAMERA_PROP) == rig.name), None)


def build_camera(rig, track, context, video=None, first_frame=0, name=None):
    """Create or refresh `rig`'s video camera from a CameraTrack. Returns the object."""
    scene = context.scene
    cam = camera_of(rig)
    if cam is None:
        name = name or "%s-camera" % rig.name
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        for coll in (rig.users_collection or (scene.collection,)):
            coll.objects.link(cam)
        cam[CAMERA_PROP] = rig.name
    cam.parent = rig
    cam.matrix_parent_inverse.identity()
    cam.rotation_mode = "QUATERNION"

    data = cam.data
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = SENSOR
    data.lens = track.fx / max(1, track.width) * SENSOR
    data.shift_x = (track.width * 0.5 - track.cx) / max(1, track.width)
    data.shift_y = (track.cy - track.height * 0.5) / max(1, track.width)
    data.clip_start = 0.05
    scene.render.resolution_x = track.width
    scene.render.resolution_y = track.height
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0

    _key(cam, track)
    _background(data, video, first_frame)
    scene.camera = cam
    return cam


def _key(cam, track):
    ad = cam.animation_data or cam.animation_data_create()
    old = ad.action
    action = bpy.data.actions.new(cam.name)
    ad.action = action
    _assign_slot(cam, action)
    if old is not None and old.users == 0:
        bpy.data.actions.remove(old)
    frames = range(len(track.location))
    for prop, rows in (("location", track.location), ("rotation_quaternion", track.rotation)):
        for axis in range(len(rows[0])):
            fc = _curve(action, prop, axis, "Camera")
            fc.keyframe_points.add(len(rows))
            fc.keyframe_points.foreach_set(
                "co", [v for f in frames for v in (float(f), rows[f][axis])])
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()


def _background(data, video, first_frame):
    """The video behind the camera view, frame for frame: GEM-X keeps every frame of the
    input, so video frame i is result frame i, and scene frame k shows frame first + k."""
    for bg in list(data.background_images):
        data.background_images.remove(bg)
    if not video:
        data.show_background_images = False
        return
    clip = bpy.data.movieclips.load(video, check_existing=True)
    clip.frame_start = -int(first_frame)
    bg = data.background_images.new()
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    bg.alpha = 1.0
    bg.display_depth = "BACK"
    bg.frame_method = "FIT"
    data.show_background_images = True
