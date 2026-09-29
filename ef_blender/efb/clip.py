"""Blender action <-> Epic Fight animation json, without bpy.

The bpy shell reads the scene into a PoseTable and hands it here; everything below is
plain arithmetic on Mat4 and lists, so it runs and is tested in ordinary python.

Two things in this file are measured facts about the shipped data, not conventions we
picked (see anim_gate.py, which re-derives both from the jar):

  * every transform number in every shipped animation json is round(v, 6);
  * every timestamp is round(t, 4), and t = round(frame / fps, 4) with fps = 60 for
    376 of the 383 clips, 24 or 120 for the rest. One file, biped/living/equip_item.json,
    still carries the exporter's leftover "fps": 60.0 key, which the loader ignores.

So export quantises to those grids. Anything else diffs dirty against upstream on every
single line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .animjson import ATTRIBUTES, MATRIX, AnimationDocument, TRS, Track
from .matrix import Mat4, quat_normalize
from .rig import COORD_BONE, Rig

__all__ = [
    "EF_TIME_DECIMALS", "EF_VALUE_DECIMALS", "FPS_CANDIDATES",
    "KEYED", "BAKED", "AUTO", "AUTO_DRIFT",
    "JOINTS_SOURCE", "JOINTS_ALL", "JOINTS_KEYED", "export_joints",
    "time_from_frame", "frame_from_time", "snap_frame", "frames_for_times",
    "suggest_fps", "quantize", "quantize_matrix", "quantize_trs",
    "PoseTable", "ClipMeta", "ImportedClip", "SHEAR_VISIBLE",
    "plan_frames", "build_document", "read_document", "interpolation_error", "root_first",
    "document_fps", "hemisphere_continuous", "pose_table_from_clip",
]

EF_TIME_DECIMALS = 4
EF_VALUE_DECIMALS = 6

SHEAR_VISIBLE = 1e-4

FPS_CANDIDATES = (60, 30, 24, 25, 20, 50, 48, 120, 15, 12, 10)

KEYED = "keyed"
BAKED = "baked"
AUTO = "auto"

AUTO_DRIFT = 0.005

JOINTS_SOURCE = "SOURCE"
JOINTS_ALL = "ALL"
JOINTS_KEYED = "KEYED"


def export_joints(rig: Rig, keys, coverage=JOINTS_ALL, order=()):
    """Which joints get a track, hierarchy order, Coord first when it is animated.

    ALL is the default and is what the game needs. A joint the file leaves out is not
    skipped at playback - Armature.getPoseTransform reads Pose.orElseEmpty and gets
    JointTransform.empty, i.e. the bind pose - so an absent Shoulder_R pins the parent of
    the whole right arm while the arm animates under it, and an absent Torso/Chest pins
    the upper body while the legs move. It also costs nothing: bakeKeyframes unions every
    sheet's timestamps anyway, and setLinkAnimation builds its joint set from the two
    poses, so a joint missing from the target snaps to bind across a transition instead of
    blending.

    SOURCE reproduces the track list of the file an action was imported from, plus
    anything keyed since - that is for re-exporting a clip that has to diff clean against
    upstream, not for authoring, and it will reproduce a broken source faithfully. KEYED
    writes only what was touched, and closes the chain up to the root: a joint between two
    that are written is never dropped, see _with_ancestors.

    The armature root is emitted whatever the coverage says. JsonAssetLoader hands
    mulFront(BLENDER_TO_MINECRAFT_COORD) to the first track that resolves to a joint, so a
    file without Root gives that correction to whichever joint happens to come first and
    the whole clip arrives rotated 90 degrees.

    Coord is not an armature joint and stays keyed-driven whatever the coverage: a Coord
    track changes how JsonAssetLoader reads Root, so inventing one would change the clip.
    """
    deform = rig.deform_bones()
    if coverage == JOINTS_ALL:
        want = set(deform)
    elif coverage == JOINTS_KEYED:
        want = _with_ancestors(rig, {n for n in deform if keys.get(n)})
    elif coverage == JOINTS_SOURCE:
        source = {n for n in (order or ()) if n in rig}
        want = set(deform) if not source else source | {n for n in deform if keys.get(n)}
    else:
        raise ValueError("unknown joint coverage %r" % (coverage,))
    want.add(rig.armature.root)
    head = [COORD_BONE] if rig.has_coord and keys.get(COORD_BONE) else []
    return head + [n for n in deform if n in want]


def _with_ancestors(rig: Rig, want):
    """Close a joint set upwards, so no emitted joint has an unemitted ancestor.

    A skipped joint is worse than pinned. A track is written against the nearest joint the
    file also carries, and Epic Fight divides every track by that joint's own rest instead,
    so skipping Shoulder_R folds the whole spine offset into Arm_R and the game takes it
    back out at the wrong place. Measured on the reported 13-joint clip: the legs and spine
    land exactly, the right arm 0.39 blocks out, the hand 0.58, the tool 0.85.

    Only leaves are ever safe to drop, which is also all upstream drops - its 32 partial
    clips omit the legs, the head and the seam markers and nothing in between. Coord stops
    the walk: it is not an armature joint, and inventing a Coord track changes how
    JsonAssetLoader reads Root.
    """
    out = set(want)
    for name in want:
        cur = rig.parent(name)
        while cur is not None and cur != COORD_BONE:
            out.add(cur)
            cur = rig.parent(cur)
    return out



def time_from_frame(frame, fps, decimals=EF_TIME_DECIMALS) -> float:
    return round(float(frame) / float(fps), decimals)


def frame_from_time(time, fps) -> float:
    return float(time) * float(fps)


def snap_frame(time, fps, decimals=EF_TIME_DECIMALS) -> float:
    """The frame to place `time` on: the nearest integer when that reproduces the
    timestamp exactly, otherwise the exact fractional frame (which still round-trips,
    because the 4-decimal rounding swallows the error)."""
    raw = frame_from_time(time, fps)
    near = float(round(raw))
    if time_from_frame(near, fps, decimals) == time:
        return near
    return raw


def frames_for_times(times, fps, decimals=EF_TIME_DECIMALS):
    """(frames, lossy) - lossy lists the timestamps no frame reproduces at this fps."""
    frames, lossy = [], []
    for t in times:
        f = snap_frame(t, fps, decimals)
        if time_from_frame(f, fps, decimals) != t:
            lossy.append(t)
        frames.append(f)
    return frames, lossy


def suggest_fps(times, candidates=FPS_CANDIDATES, decimals=EF_TIME_DECIMALS):
    """The first candidate rate that puts every timestamp on a whole frame, else None."""
    for fps in candidates:
        if all(time_from_frame(round(frame_from_time(t, fps)), fps, decimals) == t
               for t in times):
            return fps
    return None


def document_fps(doc: AnimationDocument, candidates=FPS_CANDIDATES, fallback=60):
    times = set()
    for t in doc.tracks:
        times.update(t.times)
    if doc.camera:
        times.update(doc.camera.times)
    return suggest_fps(sorted(times), candidates) or fallback



def quantize(v, decimals=EF_VALUE_DECIMALS) -> float:
    return round(float(v), decimals)


def quantize_matrix(m: Mat4, decimals=EF_VALUE_DECIMALS) -> Mat4:
    return Mat4([round(v, decimals) for v in m.flat()])


def quantize_trs(t: TRS, decimals=EF_VALUE_DECIMALS) -> TRS:
    return TRS(tuple(round(v, decimals) for v in t.loc),
               tuple(round(v, decimals) for v in t.rot),
               tuple(round(v, decimals) for v in t.sca))



@dataclass
class PoseTable:
    """One evaluation of the rig over a set of frames.

    frames    ascending, the union of every frame anything is sampled at
    basis     bone -> pose_bone.matrix_basis at each frame
    world     bone -> pose_bone.matrix (armature space) at each frame
    channels  bone -> (location, rotation_quaternion, scale) at each frame
    keyed     bone -> the subset of `frames` that bone emits a key on

    `channels` is not redundant: decomposing `basis` throws away the quaternion's sign,
    which an ATTRIBUTES-format file writes out and would then read back negated.
    """
    frames: list = field(default_factory=list)
    basis: dict = field(default_factory=dict)
    world: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)
    keyed: dict = field(default_factory=dict)
    _ix: dict = field(default_factory=dict, repr=False)

    def index(self, frame) -> int:
        if len(self._ix) != len(self.frames):
            self._ix = {f: i for i, f in enumerate(self.frames)}
        return self._ix[frame]

    def bones(self):
        return list(self.basis)


@dataclass
class ClipMeta:
    """Everything an Epic Fight json carries that a Blender action has no slot for.

    Kept so a re-export reproduces the original file: the top-level key order, the
    declared format, the sidecar blocks (visibilities, priority, ...) and any track
    naming a joint the rig does not have.
    """
    fps: float = 60.0
    format: "str | None" = None
    order: tuple = ()
    key_order: tuple = ()
    extras: dict = field(default_factory=dict)
    foreign: dict = field(default_factory=dict)
    camera: "object | None" = None
    source: str = ""

    @classmethod
    def from_document(cls, doc: AnimationDocument, fps, source="") -> "ClipMeta":
        return cls(fps=fps, format=doc.format, order=tuple(doc.names),
                   key_order=tuple(doc.key_order), extras=dict(doc.extras),
                   camera=doc.camera, source=source)


@dataclass
class ImportedClip:
    """What the importer needs to key onto the rig.

    `channels` is the payload: (loc, quat_wxyz, scale) per key, which is literally what a
    pose bone stores. An ATTRIBUTES track hands its numbers over untouched - decomposing
    and recomposing them would lose the quaternion's sign, which is free information the
    file already carries.
    """
    frames: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)
    basis: dict = field(default_factory=dict)
    meta: ClipMeta = field(default_factory=ClipMeta)
    lossy: list = field(default_factory=list)
    unrepresentable: dict = field(default_factory=dict)
    shear_metres: dict = field(default_factory=dict)



def plan_frames(mode, keys_by_bone, frame_range, step=1.0, bones=None) -> dict:
    """Which frames each bone emits a key on.

    KEYED - the bone's own keyframes, which is what upstream did: 339 of the 383 shipped
    clips give different joints different time arrays, and walk.json is 7 authored keys
    over 0.783 s rather than a 47-frame bake. A bone with fewer than two keys in range
    gets the two ends, because every shipped track has at least two.

    BAKED - one shared ladder over the range at `step` frames. Needed when the pose comes
    from anything Epic Fight cannot re-evaluate: bezier or constant interpolation,
    constraints, IK, drivers. Epic Fight only lerps translation and scale and nlerps
    rotation between the keys it is given.
    """
    first, last = float(frame_range[0]), float(frame_range[1])
    names = list(bones if bones is not None else keys_by_bone)
    if mode == BAKED:
        ladder = _ladder(first, last, step)
        return {n: list(ladder) for n in names}
    if mode != KEYED:
        raise ValueError("unknown sampling mode %r" % (mode,))
    out = {}
    for n in names:
        got = sorted({float(f) for f in keys_by_bone.get(n, ())
                      if first - 1e-9 <= f <= last + 1e-9})
        out[n] = got if len(got) >= 2 else ([first, last] if last > first else [first])
    return out


def _ladder(first, last, step):
    if step <= 0.0:
        raise ValueError("bake step must be positive")
    out, f, i = [], first, 0
    while f <= last + 1e-9:
        out.append(f)
        i += 1
        f = first + i * step
    if out and out[-1] < last - 1e-9:
        out.append(last)
    return out



def build_document(rig: Rig, table: PoseTable, fps, *, order=None, format=None,
                   meta: "ClipMeta | None" = None,
                   time_decimals=EF_TIME_DECIMALS,
                   value_decimals=EF_VALUE_DECIMALS) -> AnimationDocument:
    """A PoseTable becomes an Epic Fight animation document.

    `order` is the output track order; it defaults to the rig hierarchy, which is what
    every shipped file uses, and Coord goes first when it is emitted. Tracks named in
    meta.foreign are re-emitted verbatim - they name joints no armature has. Passing an
    order of your own skips root_first, so put the rig root ahead of every other joint.
    """
    meta = meta or ClipMeta(fps=fps)
    fmt = format if format is not None else meta.format
    order = list(order) if order is not None else _default_order(rig, table, meta)
    emitted = [n for n in order if n not in meta.foreign]
    parents = rig.emitted_parents(emitted)

    tracks = []
    for name in order:
        if name in meta.foreign:
            tracks.append(meta.foreign[name])
            continue
        frames = table.keyed.get(name) or []
        times, keys = [], []
        for f in frames:
            t = time_from_frame(f, fps, time_decimals)
            if times and t <= times[-1]:
                continue
            i = table.index(f)
            times.append(t)
            if fmt == ATTRIBUTES:
                keys.append(quantize_trs(_export_trs(rig, table, name, parents[name], i),
                                         value_decimals))
            else:
                value = _export_value(rig, table, name, parents[name], i)
                keys.append(quantize_matrix(value, value_decimals))
        tracks.append(Track(name, times, keys, ATTRIBUTES if fmt == ATTRIBUTES else MATRIX))

    doc = AnimationDocument(tracks, meta.camera, fmt, dict(meta.extras),
                            tuple(meta.key_order))
    return doc


def _default_order(rig: Rig, table: PoseTable, meta: ClipMeta):
    keyed = {n for n, f in table.keyed.items() if f}
    order = [n for n in rig.bones() if n in keyed]
    if meta.order:
        known = set(order) | set(meta.foreign)
        first = [n for n in meta.order if n in known]
        order = first + [n for n in order if n not in set(first)]
    else:
        order += [n for n in meta.foreign if n not in set(order)]
    return root_first(rig, order)


def root_first(rig: Rig, order):
    """Put the rig root ahead of every other track the armature can resolve.

    JsonAssetLoader carries a `root` flag that is cleared by the first entry naming a
    joint it knows, and only that entry gets mulFront(BLENDER_TO_MINECRAFT_COORD). Any
    other joint there rotates the whole clip 90 degrees. Tracks the armature does not
    know are left where they are - they never clear the flag, which is why the one shipped
    clip that opens with IK_left still loads right.
    """
    want = COORD_BONE if COORD_BONE in order and rig.has_coord else rig.armature.root
    if want not in order:
        return order
    at = next((i for i, n in enumerate(order) if n in rig), None)
    if at is None or order[at] == want:
        return order
    rest = [n for n in order if n != want]
    return rest[:at] + [want] + rest[at:]


def _export_value(rig: Rig, table: PoseTable, name, parent, i) -> Mat4:
    """The matrix a Blender exporter writes for `name`: its pose relative to `parent`.

    When the emitted parent is the rig parent this is rest_local @ matrix_basis, which is
    exact and independent of anything the parent is doing. It only has to go through the
    armature-space matrices when a bone's rig parent was left out of the export - Root
    under an unemitted Coord - and then the parent's motion is folded in on purpose.
    """
    if parent == rig.parent(name):
        return rig.rest_local(name) @ table.basis[name][i]
    world = table.world[name][i]
    return world if parent is None else table.world[parent][i].inverse() @ world


def _unit_quat(quat, decimals=EF_VALUE_DECIMALS):
    """JsonAssetLoader's ATTRIBUTES branch never normalises, unlike the MATRIX one, so a
    short quaternion scales the joint in game. Blender normalises on the way into
    matrix_basis, so the unit one is the pose the artist sees. A baked frame is a raw
    lerp of two keys and can be several percent short; a key straight off a rounded file
    is short by less than the 6th decimal, and churning that would only dirty the diff.
    """
    unit = quat_normalize(quat)
    if max(abs(a - b) for a, b in zip(unit, quat)) <= 10.0 ** -decimals:
        return tuple(quat)
    return unit


def _export_trs(rig: Rig, table: PoseTable, name, parent, i) -> TRS:
    """The ATTRIBUTES payload: the pose channels themselves when the emitted parent is
    the rig parent, which keeps the quaternion's sign that a decomposition would drop."""
    chan = table.channels.get(name)
    if chan and parent == rig.parent(name):
        loc, quat, scale = chan[i]
        return TRS(tuple(loc), _unit_quat(quat), tuple(scale))
    rest = rig.rest_between(name, parent)
    return TRS.from_matrix(rest.inverse() @ _export_value(rig, table, name, parent, i))



def hemisphere_continuous(channels):
    """Flip q -> -q so consecutive keys stay on the same side of the quaternion sphere.

    Epic Fight's MathUtils.lerpQuaternion negates the far key when the dot product is
    negative, so it always takes the short arc. A LINEAR fcurve does not: Blender lerps
    the stored components as they are and spins the long way round. Without this the
    imported action plays differently from the game between exactly those two keys.
    """
    out = []
    prev = None
    for loc, quat, scale in channels:
        if prev is not None and sum(a * b for a, b in zip(prev, quat)) < 0.0:
            quat = tuple(-v for v in quat)
        prev = quat
        out.append((loc, quat, scale))
    return out


def _tip_gap(raw: Mat4, landed: Mat4, radius: float) -> float:
    """Metres a point `radius` out on the joint's own axes moves between the two.

    The shear the file asks for and the pose Blender can hold differ by a linear map, so
    this is the whole of what the import loses, measured where it is largest.
    """
    worst = 0.0
    for p in ((radius, 0.0, 0.0), (0.0, radius, 0.0), (0.0, 0.0, radius)):
        a, b = raw.transform_point(p), landed.transform_point(p)
        worst = max(worst, sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)
    return worst


def read_document(rig: Rig, doc: AnimationDocument, fps, *,
                  time_decimals=EF_TIME_DECIMALS, source="",
                  hemisphere=True) -> ImportedClip:
    """An Epic Fight document becomes per-bone matrix_basis keys on the rig.

    matrix_basis is the value Epic Fight keeps in its TransformSheet, so this is the same
    arithmetic the game does, just landing in Blender instead of a Pose.
    """
    meta = ClipMeta.from_document(doc, fps, source)
    known = [t.name for t in doc.tracks if t.name in rig]
    parents = rig.emitted_parents(known)

    clip = ImportedClip(meta=meta)
    for track in doc.tracks:
        if track.name not in rig:
            meta.foreign[track.name] = track
            continue
        rest = rig.rest_between(track.name, parents[track.name])
        frames, lossy = frames_for_times(track.times, fps, time_decimals)
        chan, basis = [], []
        for i in range(len(track)):
            key = track.keys[i]
            if isinstance(key, TRS):
                chan.append((key.loc, key.rot, key.sca))
                basis.append(key.to_matrix())
            else:
                delta = track.delta_at(i, rest)
                chan.append(delta.decompose())
                basis.append(delta)
        order = sorted(range(len(frames)), key=lambda i: frames[i])
        chan = [chan[i] for i in order]
        clip.frames[track.name] = [frames[i] for i in order]
        clip.channels[track.name] = hemisphere_continuous(chan) if hemisphere else chan
        clip.basis[track.name] = [basis[i] for i in order]
        clip.lossy += [(track.name, t) for t in lossy]
        landed = [Mat4.compose(*c) for c in clip.channels[track.name]]
        slack = max((clip.basis[track.name][i].max_abs_diff(landed[i])
                     for i in range(len(chan))), default=0.0)
        if slack > 1e-6:
            clip.unrepresentable[track.name] = slack
            clip.shear_metres[track.name] = max(
                _tip_gap(clip.basis[track.name][i], landed[i],
                         rig.bone_length(track.name)) for i in range(len(chan)))

    for name in rig.bones():
        clip.frames.setdefault(name, [])
        clip.channels.setdefault(name, [])
        clip.basis.setdefault(name, [])
    return clip



def interpolation_error(doc: AnimationDocument, rig: Rig, table: PoseTable, fps,
                        time_decimals=EF_TIME_DECIMALS):
    """How far Epic Fight's playback of the exported keys drifts from the Blender pose.

    Epic Fight lerps translation and scale and nlerps rotation between keys. Anything
    curved in Blender - bezier handles, constraints, IK - shows up here. Returns
    (max error in blocks, "bone@frame").
    """
    worst, where = 0.0, ""
    order = [t.name for t in doc.tracks]
    parents = rig.emitted_parents([n for n in order if n in rig])
    deltas = {}
    for track in doc.tracks:
        if track.name not in rig:
            continue
        rest = rig.rest_between(track.name, parents[track.name])
        deltas[track.name] = Track(track.name, list(track.times),
                                   [track.delta_at(i, rest) for i in range(len(track))],
                                   MATRIX)
    for i, frame in enumerate(table.frames):
        t = time_from_frame(frame, fps, time_decimals)
        pose = {}
        for name in rig.bones():
            src = deltas.get(name)
            parent = parents[name] if src is not None else rig.parent(name)
            base = pose[parent] if parent is not None else Mat4.identity()
            rest = rig.rest_between(name, parent)
            pose[name] = base @ rest @ (src.sample(t) if src is not None
                                        else Mat4.identity())
        for name in rig.deform_bones():
            if name not in table.world:
                continue
            a = pose[name].to_translation()
            b = table.world[name][i].to_translation()
            d = max(abs(x - y) for x, y in zip(a, b))
            if d > worst:
                worst, where = d, "%s@%g" % (name, frame)
    return worst, where


def pose_table_from_clip(rig: Rig, clip: ImportedClip) -> PoseTable:
    """What Blender would evaluate for an ImportedClip, computed without Blender.

    Channels are interpolated linearly, which is exactly what a LINEAR fcurve does - the
    quaternion included, component by component and unnormalised. Lets the whole export
    path be tested in plain python; the Blender run then only has to show that a real
    scene agrees.
    """
    frames = sorted({f for got in clip.frames.values() for f in got})
    table = PoseTable(frames=frames, keyed={n: list(clip.frames.get(n) or [])
                                            for n in rig.bones()})
    for name in rig.bones():
        got = clip.frames.get(name) or []
        chans = clip.channels.get(name) or []
        table.channels[name] = [_lerp_channels(got, chans, f) for f in frames]
        table.basis[name] = [_compose_channels(*c) for c in table.channels[name]]
    for name in rig.bones():
        parent = rig.parent(name)
        rest = rig.rest_local(name)
        base = table.world[parent] if parent else None
        table.world[name] = [
            (base[i] if base else Mat4.identity()) @ rest @ table.basis[name][i]
            for i in range(len(frames))]
    return table


def _compose_channels(loc, quat, scale) -> Mat4:
    return Mat4.compose(loc, quat_normalize(quat), scale)


_REST_CHANNELS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _lerp_channels(frames, channels, f):
    if not frames:
        return _REST_CHANNELS
    if f <= frames[0]:
        return channels[0]
    if f >= frames[-1]:
        return channels[-1]
    i = 0
    while i + 1 < len(frames) and frames[i + 1] <= f:
        i += 1
    span = frames[i + 1] - frames[i]
    t = 0.0 if span == 0.0 else (f - frames[i]) / span
    a, b = channels[i], channels[i + 1]
    return tuple(tuple(x + (y - x) * t for x, y in zip(ga, gb))
                 for ga, gb in zip(a, b))
