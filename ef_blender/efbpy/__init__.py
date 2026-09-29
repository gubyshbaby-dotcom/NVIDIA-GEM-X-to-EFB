"""The bpy shell. Everything here touches a Blender scene; the maths lives in efb.

  efbpy.build       control rig construction from efb.rigdef
  efbpy.body        the skinned body mesh, its skin material and the bind
  efbpy.snap        IK/FK snapping
  efbpy.ops         rig operators
  efbpy.spaces      which space an IK control hangs from
  efbpy.restpose    rest pose override
  efbpy.mirror      mirror a pose left to right
  efbpy.rigedit     editable rig settings and the code that applies them
  efbpy.tool        hanging a weapon object off a Tool socket
  efbpy.toolproxy   the stick that shows which way a socket points
  efbpy.ui          the N-panel
  efbpy.animscene   plain deform rig, and the bridge between Mat4 and mathutils
  efbpy.animimport  Epic Fight animation json -> action
  efbpy.ikbake      the IK handles put on an imported clip's FK result
  efbpy.animexport  action -> Epic Fight animation json
  efbpy.animops     import/export operators and their File menu entries
  efbpy.migrate     carrying an action forward from an older rig
  efbpy.rootcarry   handing an older rig's loose handles and poles to the root, in place
  efbpy.camexport   Blender cameras -> VIX camera animations plus a shot manifest

Metadata lives in blender_manifest.toml, not in a bl_info here.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from . import (animops, camexport, migrate, mirror, ops, restpose,
               rigedit, rootcarry, spaces, tool, toolproxy, ui)

_MODULES = (ops, spaces, restpose, mirror, rigedit, tool, toolproxy, ui, animops,
            migrate, rootcarry, camexport)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()
