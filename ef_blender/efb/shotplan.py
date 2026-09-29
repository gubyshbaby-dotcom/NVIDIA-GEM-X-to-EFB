"""Multi-camera shots: ranges in, a resolved cut list and a manifest out. No bpy.

VIX holds exactly one CameraAnimationFix in a static field and SetAnim replaces it
wholesale, so a shot cannot live in one file. Each camera gets its own VIX-loadable
json, zeroed at its own range start, and this manifest says who owns which stretch of
shot time. Overlaps are legal: the highest priority wins, ties go to whoever was
declared first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["CameraRange", "Segment", "Shot", "FORMAT", "dumps_manifest"]

FORMAT = "efb-vix-shot/1"
_EPS = 1e-9


@dataclass(frozen=True)
class CameraRange:
    name: str
    start: float
    end: float
    priority: int = 0
    order: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def covers(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class Segment:
    camera: str
    start: float
    end: float
    offset: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Shot:
    name: str
    ranges: list
    origin_mode: str = "entity"
    origin_object: str = ""
    fps: float = 0.0
    warnings: list = field(default_factory=list)

    def __post_init__(self):
        fixed = []
        for i, r in enumerate(self.ranges):
            if r.end - r.start <= _EPS:
                raise ValueError("camera %r has an empty range" % r.name)
            fixed.append(CameraRange(r.name, r.start, r.end, r.priority, i))
        self.ranges = sorted(fixed, key=lambda r: (r.start, r.order))
        names = [r.name for r in self.ranges]
        if len(set(names)) != len(names):
            raise ValueError("duplicate camera names: %r" % names)

    @property
    def start(self) -> float:
        return min(r.start for r in self.ranges)

    @property
    def duration(self) -> float:
        return max(r.end for r in self.ranges)

    def by_name(self, name) -> CameraRange:
        for r in self.ranges:
            if r.name == name:
                return r
        raise KeyError(name)

    def resolve(self):
        """Flatten the ranges into a non-overlapping cut list."""
        bounds = sorted({b for r in self.ranges for b in (r.start, r.end)})
        out = []
        for lo, hi in zip(bounds, bounds[1:]):
            if hi - lo <= _EPS:
                continue
            mid = 0.5 * (lo + hi)
            live = [r for r in self.ranges if r.covers(mid)]
            if not live:
                continue
            win = max(live, key=lambda r: (r.priority, -r.order))
            if out and out[-1].camera == win.name and abs(out[-1].end - lo) <= _EPS:
                out[-1] = Segment(win.name, out[-1].start, hi, out[-1].offset)
            else:
                out.append(Segment(win.name, lo, hi, lo - win.start))
        return out

    def gaps(self):
        covered, out, cursor = self.resolve(), [], self.start
        for seg in covered:
            if seg.start - cursor > _EPS:
                out.append((cursor, seg.start))
            cursor = seg.end
        if self.duration - cursor > _EPS:
            out.append((cursor, self.duration))
        return out

    def check(self):
        """Everything the animator should know before shipping the shot."""
        msgs = []
        for lo, hi in self.gaps():
            msgs.append("gap %.4f..%.4f s: VIX blends back to the gameplay camera there"
                        % (lo, hi))
        used = {s.camera for s in self.resolve()}
        for r in self.ranges:
            if r.name not in used:
                msgs.append("camera %r is fully covered by higher priority ranges" % r.name)
        for s in self.resolve():
            if s.offset > _EPS:
                msgs.append("camera %r resumes at %.4f s into its own clip; the runtime "
                            "must sample it at shotTime - range start" % (s.camera, s.offset))
        return msgs

    def manifest(self, files=None) -> dict:
        files = files or {}
        segs = self.resolve()
        return {
            "format": FORMAT,
            "name": self.name,
            "duration": round(self.duration, 6),
            "fps": self.fps,
            "origin": {"mode": self.origin_mode,
                       "object": self.origin_object,
                       "lock_org_pos": self.origin_mode != "entity",
                       "dynamic_y": self.origin_mode == "locked_dynamic_y"},
            "cameras": [{"name": r.name,
                         "file": files.get(r.name, r.name + ".json"),
                         "start": round(r.start, 6),
                         "end": round(r.end, 6),
                         "duration": round(r.duration, 6),
                         "priority": r.priority} for r in self.ranges],
            "timeline": [{"start": round(s.start, 6),
                          "end": round(s.end, 6),
                          "camera": s.camera,
                          "offset": round(s.offset, 6)} for s in segs],
            "gaps": [[round(a, 6), round(b, 6)] for a, b in self.gaps()],
            "warnings": list(self.warnings) + self.check(),
        }


def dumps_manifest(shot: Shot, files=None, indent=4, newline="\r\n") -> bytes:
    text = json.dumps(shot.manifest(files), indent=indent)
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")
