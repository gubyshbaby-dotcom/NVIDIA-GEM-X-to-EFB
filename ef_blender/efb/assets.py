"""Where the armature and the body mesh come from: the bundled copy, or a jar.

One rule, in one place. A jar path that really exists wins, so anyone pointing at a
modded armature keeps the old behaviour; otherwise the bundled data answers, and only an
entity nobody bundles falls through to the jar and its error message.
"""

from __future__ import annotations

import os

from . import bundled
from .jarsource import EfJar

__all__ = ["source_of", "armature", "body_mesh", "skin_path"]


def source_of(entity=bundled.DEFAULT_VARIANT, jar_path=None) -> str:
    """"jar" or "bundled" - which one a call with these arguments will read."""
    if jar_path and os.path.isfile(jar_path):
        return "jar"
    return "bundled" if bundled.available(entity) else "jar"


def armature(entity=bundled.DEFAULT_VARIANT, jar_path=None):
    if source_of(entity, jar_path) == "bundled":
        return bundled.armature(entity)
    with EfJar(jar_path or None) as jar:
        return jar.armature(entity)


def body_mesh(entity=bundled.DEFAULT_VARIANT, jar_path=None):
    if source_of(entity, jar_path) == "bundled":
        return bundled.body_mesh(entity)
    with EfJar(jar_path or None) as jar:
        return jar.body_mesh(entity)


def skin_path(entity=bundled.DEFAULT_VARIANT):
    """The default skin. It is never in the mod jar, so this is always the bundled one."""
    return bundled.skin_path(entity) if bundled.available(entity) else ""
