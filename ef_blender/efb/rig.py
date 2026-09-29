"""The Blender-side skeleton: an Epic Fight armature plus the optional Coord bone.

Coord is the root-motion bone. It never appears in an armature json - JsonAssetLoader
strips it - but 43 of the shipped clips animate it, and in those files the Root track is
written relative to it. Its rest orientation is Rx(+90), the inverse of the
Blender->Minecraft basis, which is why a Root track sitting under Coord arrives already
in Minecraft orientation and upstream then skips the root correction on it.

A Rig carries no bpy. It answers three questions the exporter and importer both need:
which bones exist, what each bone's rest matrix is, and - given the set of joints a file
actually carries - which bone each track is written relative to.
"""

from __future__ import annotations

from .armature import COORD_BONE
from .coords import MINECRAFT_TO_BLENDER
from .matrix import Mat4

__all__ = ["Rig", "COORD_REST", "COORD_BONE"]

COORD_REST = MINECRAFT_TO_BLENDER

_DEFAULT_LENGTH = 0.1


class Rig:
    """An armature plus, optionally, a Coord bone parented above its root."""

    def __init__(self, armature, coord: bool = True):
        self.armature = armature
        self.has_coord = bool(coord)
        self._order = ([COORD_BONE] if self.has_coord else []) + armature.depth_first()
        self._deform = tuple(armature.depth_first())


    @property
    def root(self) -> str:
        return COORD_BONE if self.has_coord else self.armature.root

    def __contains__(self, name) -> bool:
        if name == COORD_BONE:
            return self.has_coord
        return name in self.armature

    def __len__(self):
        return len(self._order)

    def bones(self):
        """Every bone, hierarchy order, Coord first when present."""
        return list(self._order)

    def deform_bones(self):
        """The Epic Fight joints, hierarchy order. Coord is not one of them."""
        return list(self._deform)

    def children(self, name):
        if name == COORD_BONE:
            return [self.armature.root]
        return list(self.armature[name].children)

    def parent(self, name):
        if name == COORD_BONE:
            return None
        if name == self.armature.root:
            return COORD_BONE if self.has_coord else None
        return self.armature[name].parent


    def rest_model(self, name) -> Mat4:
        """Armature-space rest matrix, Blender frame."""
        if name == COORD_BONE:
            return COORD_REST
        return self.armature.rest_model(name)

    def rest_local(self, name) -> Mat4:
        """Rest matrix relative to this bone's rig parent."""
        return self.rest_between(name, self.parent(name))

    def rest_between(self, name, parent) -> Mat4:
        """Rest of `name` relative to `parent`; armature space when parent is None."""
        if parent is None:
            return self.rest_model(name)
        if parent == self.parent(name) and name != self.armature.root:
            return self.armature[name].rest_local
        return self.rest_model(parent).inverse() @ self.rest_model(name)

    def bone_length(self, name) -> float:
        """Cosmetic. Blender needs a nonzero length; it never enters matrix_local."""
        best = 0.0
        for c in self.children(name):
            best = max(best, self.rest_between(c, name).to_translation()[1])
        return best if best > 1e-5 else _DEFAULT_LENGTH


    def emitted_parents(self, emitted) -> dict:
        """Map each emitted bone to the nearest ancestor that is also emitted, or None.

        A track is written relative to that bone, exactly as Blender exports a bone whose
        own parent was left out of the selection.
        """
        present = set(emitted)
        out = {}
        for name in emitted:
            p = self.parent(name)
            while p is not None and p not in present:
                p = self.parent(p)
            out[name] = p
        return out

    def root_correction_target(self, order):
        """The track JsonAssetLoader hands BLENDER_TO_MINECRAFT to, or None.

        A Coord track takes it unconditionally and denies it to everything after; failing
        that, the first track naming a joint the armature knows takes it.
        """
        for name in order:
            if name == COORD_BONE or name in self.armature:
                return name
        return None
