# Test bootstrap: make the addon package importable both inside and outside
# Blender. Outside Blender we inject a minimal stub for the `bpy` module so
# pure-logic units (protocol packing, camera/light math, color helpers) can be
# unit-tested without a Blender installation (e.g. in CI).

import importlib
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON_DIR = os.path.join(REPO_ROOT, "addon")


def install_fake_bpy():
    """Insert stub `bpy` modules unless a real one is already available.

    Registers bpy.types / bpy.props / bpy.utils as importable submodules so
    `from bpy.types import RenderEngine` works outside Blender."""
    if "bpy" in sys.modules:
        return sys.modules["bpy"]
    try:
        import bpy  # noqa: F401  (real Blender runtime)
        return sys.modules["bpy"]
    except ImportError:
        pass

    def _mod(name):
        m = types.ModuleType(name)
        m.__path__ = []          # mark as package so 'from x.y import z' works
        sys.modules[name] = m
        return m

    bpy = _mod("bpy")
    t = _mod("bpy.types")
    p = _mod("bpy.props")
    u = _mod("bpy.utils")

    class RenderEngine:
        pass

    class PropertyGroup:
        pass

    class Panel:
        pass

    def _prop_factory(**_kw):
        return None

    t.RenderEngine = RenderEngine
    t.PropertyGroup = PropertyGroup
    t.Panel = Panel
    p.BoolProperty = _prop_factory
    p.FloatProperty = _prop_factory
    p.IntProperty = _prop_factory
    p.PointerProperty = _prop_factory
    u.register_class = lambda cls: None
    u.unregister_class = lambda cls: None

    bpy.types = t
    bpy.props = p
    bpy.utils = u
    return bpy


def load_addon():
    """Import the addon package ('addon') with bpy satisfied.

    Also eagerly imports the lazily-loaded sibling modules (scene_shm, shm,
    core_proc) so tests can reach them as attributes of the package."""
    install_fake_bpy()
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    mod = importlib.import_module("addon")
    for sub in ("scene_shm", "shm", "core_proc"):
        try:
            importlib.import_module("addon." + sub)
        except ImportError:
            pass
    return mod
