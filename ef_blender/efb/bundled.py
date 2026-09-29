"""The armature, body mesh and default skin that ship inside the addon.

Everything here lives in efb/data, so Generate produces a textured character with no jar
path, no second .blend and no network. The two jsons are the mod's own bytes, byte for
byte - the 20 deform rest matrices have to stay identical to what the exporter writes
back, so they are copied, never regenerated.
"""

from __future__ import annotations

import os

from .armature import Armature
from .meshjson import load_mesh

__all__ = ["DATA_DIR", "VARIANTS", "DEFAULT_VARIANT", "model_path", "model_bytes",
           "skin_path", "armature", "body_mesh", "available"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

VARIANTS = {
    "biped": ("biped.json", "steve.png"),
    "biped_slim_arm": ("biped_slim_arm.json", "alex.png"),
}
DEFAULT_VARIANT = "biped"


def _entry(variant):
    try:
        return VARIANTS[variant or DEFAULT_VARIANT]
    except KeyError:
        raise ValueError("no bundled data for %r; have %s"
                         % (variant, ", ".join(sorted(VARIANTS)))) from None


def model_path(variant=DEFAULT_VARIANT) -> str:
    return os.path.join(DATA_DIR, _entry(variant)[0])


def skin_path(variant=DEFAULT_VARIANT) -> str:
    return os.path.join(DATA_DIR, _entry(variant)[1])


def model_bytes(variant=DEFAULT_VARIANT) -> bytes:
    with open(model_path(variant), "rb") as fh:
        return fh.read()


def armature(variant=DEFAULT_VARIANT) -> Armature:
    return Armature.from_bytes(model_bytes(variant), variant or DEFAULT_VARIANT)


def body_mesh(variant=DEFAULT_VARIANT):
    return load_mesh(model_bytes(variant), variant or DEFAULT_VARIANT)


def available(variant=DEFAULT_VARIANT) -> bool:
    e = VARIANTS.get(variant or DEFAULT_VARIANT)
    return bool(e) and all(os.path.isfile(os.path.join(DATA_DIR, f)) for f in e)
