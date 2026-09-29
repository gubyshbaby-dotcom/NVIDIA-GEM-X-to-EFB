"""Epic Fight animation json: read, model, write byte-exact.

Two on-disk keyframe formats, decided by the top-level "format" string and then
overridden per value by JsonAssetLoader if the shape disagrees:

  MATRIX      transform[i] is 16 floats, row-major, and holds the joint's
              PARENT-RELATIVE POSE matrix (rest included).
  ATTRIBUTES  transform[i] is {"loc","rot","sca"} and holds the joint's LOCAL DELTA
              only - rest excluded. "rot" is [w, x, y, z].

The two are related by  matrix_form = rest_local @ attributes_form, so the delta Epic
Fight ends up storing is the same either way. See Track.delta_at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .armature import COORD_BONE
from .matrix import Mat4, quat_normalize
from .coords import BLENDER_TO_MINECRAFT

__all__ = ["TRS", "Track", "CameraTrack", "AnimationDocument",
           "MATRIX", "ATTRIBUTES", "loads", "dumps", "load", "save",
           "json_loads", "json_dumps"]

MATRIX = "matrix"
ATTRIBUTES = "attributes"


@dataclass(frozen=True)
class TRS:
    """loc / rot / sca exactly as the json spells them. rot is (w, x, y, z)."""
    loc: tuple
    rot: tuple
    sca: tuple

    @classmethod
    def from_json(cls, obj) -> "TRS":
        return cls(tuple(float(v) for v in obj["loc"]),
                   tuple(float(v) for v in obj["rot"]),
                   tuple(float(v) for v in obj["sca"]))

    @classmethod
    def from_matrix(cls, m: Mat4) -> "TRS":
        loc, rot, sca = m.decompose()
        return cls(loc, rot, sca)

    def to_json(self) -> dict:
        return {"loc": list(self.loc), "rot": list(self.rot), "sca": list(self.sca)}

    def to_matrix(self) -> Mat4:
        return Mat4.compose(self.loc, self.rot, self.sca)


@dataclass
class Track:
    """One joint's keyframes. `keys` holds Mat4 for MATRIX format, TRS for ATTRIBUTES."""
    name: str
    times: list
    keys: list
    format: str = MATRIX

    def __post_init__(self):
        if len(self.times) != len(self.keys):
            raise ValueError("track %r: %d times vs %d transforms"
                             % (self.name, len(self.times), len(self.keys)))

    def __len__(self):
        return len(self.times)

    @property
    def duration(self) -> float:
        return self.times[-1] if self.times else 0.0

    def matrix_at(self, i) -> Mat4:
        k = self.keys[i]
        return k if isinstance(k, Mat4) else k.to_matrix()

    def matrices(self):
        return [self.matrix_at(i) for i in range(len(self.keys))]

    def delta_at(self, i, rest_local: Mat4) -> Mat4:
        """The value Epic Fight keeps in its TransformSheet, and Blender's
        pose_bone.matrix_basis, whichever on-disk format this track uses."""
        if self.format == ATTRIBUTES:
            return self.matrix_at(i)
        return rest_local.inverse() @ self.matrix_at(i)

    def pose_at(self, i, rest_local: Mat4) -> Mat4:
        """Parent-relative pose matrix, whichever on-disk format this track uses."""
        if self.format == ATTRIBUTES:
            return rest_local @ self.matrix_at(i)
        return self.matrix_at(i)

    def sample(self, time: float) -> Mat4:
        """Replicates TransformSheet.getInterpolationInfo + JointTransform.interpolate:
        binary search, lerp on translation and scale, nlerp on rotation, clamped."""
        n = len(self.times)
        if n == 0:
            return Mat4.identity()
        if time < 0.0:
            time += self.times[-1]
        begin, end = 0, n - 1
        while end - begin > 1:
            i = begin + (end - begin) // 2
            if self.times[i] <= time < self.times[i + 1]:
                begin, end = i, i + 1
                break
            if self.times[i] > time:
                end = i
            elif self.times[i + 1] <= time:
                begin = i
        span = self.times[end] - self.times[begin]
        t = 1.0 if span == 0.0 else min(1.0, max(0.0, (time - self.times[begin]) / span))
        a, b = self.matrix_at(begin).decompose(), self.matrix_at(end).decompose()
        loc = tuple(x + (y - x) * t for x, y in zip(a[0], b[0]))
        sca = tuple(x + (y - x) * t for x, y in zip(a[2], b[2]))
        qa, qb = a[1], b[1]
        if sum(x * y for x, y in zip(qa, qb)) < 0.0:
            qb = tuple(-v for v in qb)
        rot = quat_normalize(tuple(x * (1.0 - t) + y * t for x, y in zip(qa, qb)))
        return Mat4.compose(loc, rot, sca)


    @classmethod
    def from_json(cls, obj, default_format=MATRIX) -> "Track":
        times = [float(v) for v in obj["time"]]
        raw = obj["transform"]
        keys = []
        fmt = default_format
        for v in raw:
            if isinstance(v, dict):
                fmt = ATTRIBUTES
                keys.append(TRS.from_json(v))
            else:
                fmt = MATRIX
                keys.append(Mat4.from_flat(v))
        return cls(obj["name"], times, keys, fmt)

    def to_json(self) -> dict:
        return {"name": self.name,
                "time": list(self.times),
                "transform": [k.to_json() if isinstance(k, TRS) else list(k.flat())
                              for k in self.keys]}


@dataclass
class CameraTrack:
    """The optional "camera" block. AnimationSubFileReader always reads it as
    ATTRIBUTES with no root correction and no inverse-local."""
    times: list
    keys: list

    def __len__(self):
        return len(self.times)

    @property
    def duration(self) -> float:
        return self.times[-1] if self.times else 0.0

    @classmethod
    def from_json(cls, obj) -> "CameraTrack":
        return cls([float(v) for v in obj["time"]],
                   [TRS.from_json(v) for v in obj["transform"]])

    def to_json(self) -> dict:
        return {"time": list(self.times),
                "transform": [k.to_json() for k in self.keys]}


@dataclass
class AnimationDocument:
    tracks: list = field(default_factory=list)
    camera: "CameraTrack | None" = None
    format: "str | None" = None
    extras: dict = field(default_factory=dict)
    key_order: tuple = ()


    def track(self, name) -> "Track | None":
        for t in self.tracks:
            if t.name == name:
                return t
        return None

    @property
    def names(self):
        return [t.name for t in self.tracks]

    @property
    def duration(self) -> float:
        return max([t.duration for t in self.tracks], default=0.0)

    def root_correction_index(self, armature=None) -> int:
        """Index of the track JsonAssetLoader applies BLENDER_TO_MINECRAFT to.

        The loader carries a `root` flag consumed by the first track naming a known
        joint, and separately hard-codes the correction on any "Coord" track. Tracks
        naming joints the armature does not have (IK helpers) do not consume it.
        Returns -1 when no track would take it; MATRIX format only.
        """
        for i, t in enumerate(self.tracks):
            if t.name == COORD_BONE:
                return i
            if armature is None or t.name in armature:
                return i
        return -1

    def ef_deltas(self, armature) -> dict:
        """Replicates JsonAssetLoader.loadAnimationClip: joint name -> list of
        (time, delta Mat4). Unknown joint names are dropped, as upstream drops them."""
        out = {}
        root_pending = True
        for t in self.tracks:
            if t.name == COORD_BONE:
                fix = t.format == MATRIX
                out[COORD_BONE] = [
                    (t.times[i],
                     BLENDER_TO_MINECRAFT @ t.matrix_at(i) if fix else t.matrix_at(i))
                    for i in range(len(t))]
                root_pending = False
                continue
            if t.name not in armature:
                continue
            local = armature.ef_local(t.name)
            inv = local.inverse()
            rows = []
            for i in range(len(t)):
                if t.format == ATTRIBUTES:
                    rows.append((t.times[i], t.matrix_at(i)))
                    continue
                m = t.matrix_at(i)
                if root_pending:
                    m = BLENDER_TO_MINECRAFT @ m
                rows.append((t.times[i], inv @ m))
            out[t.name] = rows
            root_pending = False
        return out


    @classmethod
    def from_json(cls, obj) -> "AnimationDocument":
        fmt = obj.get("format")
        default = ATTRIBUTES if fmt == ATTRIBUTES else MATRIX
        tracks = [Track.from_json(t, default) for t in obj.get("animation", [])]
        camera = CameraTrack.from_json(obj["camera"]) if "camera" in obj else None
        extras = {k: v for k, v in obj.items()
                  if k not in ("animation", "camera", "format")}
        return cls(tracks, camera, fmt, extras, tuple(obj.keys()))

    def to_json(self) -> dict:
        built = {}
        if self.format is not None:
            built["format"] = self.format
        if self.tracks:
            built["animation"] = [t.to_json() for t in self.tracks]
        if self.camera is not None:
            built["camera"] = self.camera.to_json()
        built.update(self.extras)
        order = [k for k in self.key_order if k in built]
        order += [k for k in built if k not in order]
        return {k: built[k] for k in order}



def json_loads(data) -> dict:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def json_dumps(obj, indent=4, newline="\r\n", trailing_newline=False) -> bytes:
    """Reproduces the upstream exporter's layout: 4-space indent, CRLF, no trailing
    newline, and any array of plain numbers written inline as [a, b, c].

    Floats go through repr(), which is shortest-round-trip. That is exactly what the
    upstream files carry - all 604791 float tokens in the jar's animation files come
    back identical - so a re-export diffs clean against the originals.
    """
    out = []
    _encode(obj, 0, out, indent)
    text = "".join(out)
    if trailing_newline:
        text += "\n"
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")


def _fmt_number(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def _is_number_array(v) -> bool:
    return isinstance(v, (list, tuple)) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _encode(v, depth, out, indent):
    pad = " " * (indent * depth)
    inner = " " * (indent * (depth + 1))
    if isinstance(v, dict):
        if not v:
            out.append("{}")
            return
        items = list(v.items())
        out.append("{\n")
        for i, (k, val) in enumerate(items):
            out.append(inner)
            out.append(json.dumps(str(k)))
            out.append(": ")
            _encode(val, depth + 1, out, indent)
            out.append(",\n" if i < len(items) - 1 else "\n")
        out.append(pad + "}")
    elif isinstance(v, (list, tuple)):
        if len(v) == 0:
            out.append("[]")
            return
        if _is_number_array(v):
            out.append("[" + ", ".join(_fmt_number(x) for x in v) + "]")
            return
        out.append("[\n")
        for i, x in enumerate(v):
            out.append(inner)
            _encode(x, depth + 1, out, indent)
            out.append(",\n" if i < len(v) - 1 else "\n")
        out.append(pad + "]")
    elif isinstance(v, str):
        out.append(json.dumps(v))
    elif v is None:
        out.append("null")
    elif isinstance(v, bool):
        out.append("true" if v else "false")
    else:
        out.append(_fmt_number(v))


def loads(data) -> AnimationDocument:
    return AnimationDocument.from_json(json_loads(data))


def dumps(doc: AnimationDocument, **kw) -> bytes:
    return json_dumps(doc.to_json(), **kw)


def load(path) -> AnimationDocument:
    with open(path, "rb") as fh:
        return loads(fh.read())


def save(path, doc: AnimationDocument, **kw):
    with open(path, "wb") as fh:
        fh.write(dumps(doc, **kw))
