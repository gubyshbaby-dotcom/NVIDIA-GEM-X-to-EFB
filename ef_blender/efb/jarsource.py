"""Read Epic Fight assets straight out of the mod jar - no unpacking step."""

from __future__ import annotations

import os
import zipfile

from .animjson import AnimationDocument, json_loads
from .armature import Armature
from .meshjson import load_mesh

__all__ = ["EfJar", "DEFAULT_JAR", "ANIM_ROOT", "ENTITY_ROOT", "find_jar",
           "ARMATURE_FOR_DIR"]

ARMATURE_FOR_DIR = {
    "creeper": "creeper", "dragon": "dragon", "enderman": "enderman",
    "hoglin": "hoglin", "iron_golem": "iron_golem", "piglin": "piglin",
    "ravager": "ravager", "spider": "spider", "vex": "vex", "wither": "wither",
}

ANIM_ROOT = "assets/epicfight/animmodels/animations/"
ENTITY_ROOT = "assets/epicfight/animmodels/entity/"

DEFAULT_JAR = os.environ.get(
    "EF_JAR",
    "/home/rongus/.local/share/JentleMemes/instances/76757/mods/epicfight-21.17.3.1.jar")


def find_jar(path=None) -> str:
    path = path or DEFAULT_JAR
    if not os.path.isfile(path):
        raise FileNotFoundError("Epic Fight jar not found: %s (set EF_JAR)" % path)
    return path


class EfJar:
    def __init__(self, path=None):
        self.path = find_jar(path)
        self.zip = zipfile.ZipFile(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.zip.close()

    def read(self, name) -> bytes:
        return self.zip.read(name)

    def animation_names(self):
        """Every json under animmodels/animations, including the layer/trail sub-files."""
        return sorted(n for n in self.zip.namelist()
                      if n.startswith(ANIM_ROOT) and n.endswith(".json"))

    def clip_names(self):
        """Only the files that actually carry an "animation" array."""
        out = []
        for n in self.animation_names():
            obj = json_loads(self.read(n))
            if "animation" in obj:
                out.append(n)
        return out

    def entity_names(self):
        return sorted(n for n in self.zip.namelist()
                      if n.startswith(ENTITY_ROOT) and n.endswith(".json"))

    def armature(self, entity="biped") -> Armature:
        return Armature.from_bytes(self.entity_json(entity), entity)

    def entity_json(self, entity="biped") -> bytes:
        return self.read("%s%s.json" % (ENTITY_ROOT, entity))

    def body_mesh(self, entity="biped"):
        """The same file's `vertices` block - the skinned body the armature drives."""
        return load_mesh(self.entity_json(entity), entity)

    def entity_for(self, anim_name: str) -> str:
        """The entity an animation file's track names belong to."""
        rel = anim_name[len(ANIM_ROOT):] if anim_name.startswith(ANIM_ROOT) else anim_name
        return ARMATURE_FOR_DIR.get(rel.split("/")[0], "biped")

    def armature_for(self, anim_name: str) -> Armature:
        key = self.entity_for(anim_name)
        cache = getattr(self, "_arm_cache", None)
        if cache is None:
            cache = self._arm_cache = {}
        if key not in cache:
            cache[key] = self.armature(key)
        return cache[key]

    def clip(self, name) -> AnimationDocument:
        if not name.startswith(ANIM_ROOT):
            name = ANIM_ROOT + name.lstrip("/")
        if not name.endswith(".json"):
            name += ".json"
        return AnimationDocument.from_json(json_loads(self.read(name)))
