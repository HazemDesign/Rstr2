# Headless registration smoke test.
# Run: blender.exe --background --factory-startup --python tests/smoke_register.py
import sys

sys.path.insert(0, r"D:\blender\dev\Rstr2")

import addon_utils  # noqa: E402
import bpy  # noqa: E402

mod = addon_utils.enable("addon", default_set=True)
if mod is None:
    print("SMOKE FAIL: addon_utils.enable returned None")
    sys.exit(1)

engines = [getattr(e, "bl_idname", None) for e in bpy.types.RenderEngine.__subclasses__()]
print("Registered render engines:", engines)
if "RSTR2" not in engines:
    print("SMOKE FAIL: RSTR2 not registered")
    sys.exit(1)

print("SMOKE OK")
