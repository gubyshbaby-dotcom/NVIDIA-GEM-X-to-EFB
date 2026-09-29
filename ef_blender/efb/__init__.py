"""Pure-python ground layer for the Epic Fight / VIX Blender addon.

Nothing here imports bpy. The bpy shell reads the scene, calls into these modules and
writes files; that keeps the whole test suite runnable in plain python3.

  efb.matrix      Mat4 (row-major) and quaternion helpers
  efb.coords      the Blender <-> Minecraft basis and the Epic Fight root rule
  efb.armature    Armature / Joint, parsed from animmodels/entity/*.json
  efb.meshjson    the same file's body mesh: positions, uvs, skin weights
  efb.deform      Epic Fight's own skinning, the oracle a Blender body is checked against
  efb.bundled     the armature, mesh and skin shipped in efb/data - no jar needed
  efb.animjson    animation json read/write, byte-exact
  efb.vixcam      VIX camera json read/write and its sampler
  efb.jarsource   read all of the above straight out of the mod jar
  efb.openmatrix  verbatim port of Epic Fight's own matrix math, test oracle only
  efb.tensorio    torch.save / numpy files read without torch, numpy or pickle's trust
  efb.quat        tuple quaternions for the retarget
  efb.soma        NVIDIA SOMA: the skeleton GEM-X solves for and the motion it returns
  efb.gemx        SOMA motion -> an Epic Fight animation document for the biped
"""

from .matrix import Mat4
from .coords import BLENDER_TO_MINECRAFT, MINECRAFT_TO_BLENDER
from .armature import Armature, Joint, BIPED_GROUPS
from .meshjson import BodyMesh, load_mesh

__all__ = ["Mat4", "Armature", "Joint", "BIPED_GROUPS", "BodyMesh", "load_mesh",
           "BLENDER_TO_MINECRAFT", "MINECRAFT_TO_BLENDER"]
