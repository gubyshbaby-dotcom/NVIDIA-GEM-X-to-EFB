"""VIX camera animation json.

Confirmed against com/guhao/vix/camera/CameraAnimationFix.java (vix-port):

  {"time_scale": 1.0,
   "pos": {"time":[...], "x":[...], "y":[...], "z":[...]},
   "rot": {"time":[...], "rx":[...], "ry":[...], "rz":[...]},
   "fov": {"time":[...], "value":[...]}}

Every channel in a group re-reads that group's own "time" array, so one shared time
list per group is not a convention, it is the only thing the loader can read.
"time_scale" is optional and is a DIVISOR: TimeSheet.scaleTimes does t /= scale.

Units, from CameraEventsFix: pos is blocks in a rig-local frame that gets yaw-rotated
by (-yawLock - 90) degrees and added to the entity position; rx/ry/rz are Minecraft
pitch/yaw/roll in degrees (ry is subtracted from the locked yaw); fov is degrees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["ChannelGroup", "VixCameraAnimation", "VixPose",
           "POS_CHANNELS", "ROT_CHANNELS", "FOV_CHANNELS",
           "loads", "dumps", "load", "save"]

POS_CHANNELS = ("x", "y", "z")
ROT_CHANNELS = ("rx", "ry", "rz")
FOV_CHANNELS = ("value",)


@dataclass
class ChannelGroup:
    times: list
    channels: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, vals in self.channels.items():
            if len(vals) != len(self.times):
                raise ValueError("channel %r: %d values vs %d times"
                                 % (name, len(vals), len(self.times)))

    def __len__(self):
        return len(self.times)

    @property
    def max_time(self) -> float:
        return self.times[-1] if self.times else 0.0

    def index_at(self, time: float) -> int:
        """TimeSheet.getIndexByTime, quirks included: <=0 pins to 0, the scan uses >=,
        and running off the end clamps to the last key."""
        if time <= 0.0:
            return 0
        for i, t in enumerate(self.times):
            if t >= time:
                return max(i - 1, 0)
        return len(self.times) - 1

    def value_at(self, channel: str, time: float) -> float:
        """FloatSheet.getValueByTime: linear, clamped at the last key, and the
        1e-5 guard against a zero-length segment."""
        vals = self.channels[channel]
        i = self.index_at(time)
        if i == len(self.times) - 1:
            return vals[i]
        span = self.times[i + 1] - self.times[i]
        if span > 1e-5:
            t = (time - self.times[i]) / span
            return vals[i] * (1.0 - t) + vals[i + 1] * t
        return vals[i]

    @classmethod
    def from_json(cls, obj, names) -> "ChannelGroup":
        return cls([float(v) for v in obj["time"]],
                   {n: [float(v) for v in obj[n]] for n in names})

    def to_json(self, names) -> dict:
        out = {"time": list(self.times)}
        for n in names:
            out[n] = list(self.channels[n])
        return out


@dataclass(frozen=True)
class VixPose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    fov: float


@dataclass
class VixCameraAnimation:
    pos: ChannelGroup
    rot: ChannelGroup
    fov: ChannelGroup
    time_scale: "float | None" = None
    key_order: tuple = ()

    @property
    def scale(self) -> float:
        return 1.0 if self.time_scale is None else self.time_scale

    def scaled_times(self, group: ChannelGroup):
        s = self.scale
        return [t / s for t in group.times]

    @property
    def duration(self) -> float:
        """Longest key time after time_scale. CameraAnimationFix.totalTime leaves rz
        out of its max; with one time array per group that cannot matter, so we
        include everything."""
        return max(self.pos.max_time, self.rot.max_time, self.fov.max_time) / self.scale

    def sample(self, time: float) -> VixPose:
        """Time is in seconds after scaling, matching CameraAnimationFix.getPose."""
        s = self.scale
        pos = ChannelGroup(self.scaled_times(self.pos), self.pos.channels) if s != 1.0 else self.pos
        rot = ChannelGroup(self.scaled_times(self.rot), self.rot.channels) if s != 1.0 else self.rot
        fov = ChannelGroup(self.scaled_times(self.fov), self.fov.channels) if s != 1.0 else self.fov
        return VixPose(pos.value_at("x", time), pos.value_at("y", time),
                       pos.value_at("z", time), rot.value_at("rx", time),
                       rot.value_at("ry", time), rot.value_at("rz", time),
                       fov.value_at("value", time))

    @classmethod
    def from_json(cls, obj) -> "VixCameraAnimation":
        ts = float(obj["time_scale"]) if "time_scale" in obj else None
        return cls(ChannelGroup.from_json(obj["pos"], POS_CHANNELS),
                   ChannelGroup.from_json(obj["rot"], ROT_CHANNELS),
                   ChannelGroup.from_json(obj["fov"], FOV_CHANNELS),
                   ts, tuple(obj.keys()))

    def to_json(self) -> dict:
        built = {}
        if self.time_scale is not None:
            built["time_scale"] = self.time_scale
        built["pos"] = self.pos.to_json(POS_CHANNELS)
        built["rot"] = self.rot.to_json(ROT_CHANNELS)
        built["fov"] = self.fov.to_json(FOV_CHANNELS)
        order = [k for k in self.key_order if k in built]
        order += [k for k in built if k not in order]
        return {k: built[k] for k in order}


def loads(data) -> VixCameraAnimation:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return VixCameraAnimation.from_json(json.loads(data))


def dumps(anim: VixCameraAnimation, indent=4, newline="\r\n",
          trailing_newline=False) -> bytes:
    """Plain json.dumps(indent=4) with CRLF - the layout the shipped camera files use."""
    text = json.dumps(anim.to_json(), indent=indent)
    if trailing_newline:
        text += "\n"
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")


def load(path) -> VixCameraAnimation:
    with open(path, "rb") as fh:
        return loads(fh.read())


def save(path, anim: VixCameraAnimation, **kw):
    with open(path, "wb") as fh:
        fh.write(dumps(anim, **kw))
