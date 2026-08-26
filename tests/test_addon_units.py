# Unit tests for pure addon logic: camera math, typed-light rows, material
# and world color extraction, and core-exe path resolution. Runs with a stub
# bpy outside Blender; inside Blender it exercises the same code unchanged.

import math
import os
import types
import unittest

from util import load_addon, REPO_ROOT, ADDON_DIR

MOD = load_addon()


def identity_matrix(translation=(0.0, 0.0, 0.0), rot=None):
    """Minimal stand-in for a Blender 4x4 matrix_world: nested [row][col]
    lists supporting m[i][j], matching how the addon indexes it.
    `rot` supplies the three rotation COLUMNS (local axes in world space)."""
    axes = rot or [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    t = translation
    return [
        [float(axes[0][0]), float(axes[1][0]), float(axes[2][0]), float(t[0])],
        [float(axes[0][1]), float(axes[1][1]), float(axes[2][1]), float(t[1])],
        [float(axes[0][2]), float(axes[1][2]), float(axes[2][2]), float(t[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


class Obj:
    def __init__(self, matrix_world):
        self.matrix_world = matrix_world


class CamData:
    def __init__(self, **kw):
        d = dict(lens=50.0, sensor_fit="AUTO", sensor_width=36.0,
                 sensor_height=24.0, shift_x=0.0, shift_y=0.0)
        d.update(kw)
        self.__dict__.update(d)


class LightData:
    def __init__(self, **kw):
        d = dict(type="POINT", energy=10.0, color=(1.0, 1.0, 1.0),
                 shadow_soft_size=0.0, spot_size=0.8, spot_blend=0.15,
                 angle=0.0, size=1.0, size_y=2.0, shape="SQUARE")
        d.update(kw)
        self.__dict__.update(d)


class CameraMathTests(unittest.TestCase):
    def test_vertical_fit_uses_sensor_height(self):
        p = MOD.camera_params(CamData(sensor_fit="VERTICAL"), 16 / 9)
        self.assertAlmostEqual(p["tan_half_fov_y"], 12.0 / 50.0)

    def test_horizontal_fit_divides_by_aspect(self):
        p = MOD.camera_params(CamData(sensor_fit="HORIZONTAL"), 2.0)
        self.assertAlmostEqual(p["tan_half_fov_y"], (18.0 / 50.0) / 2.0)

    def test_auto_wide_uses_vertical_sensor(self):
        p = MOD.camera_params(CamData(), 16 / 9)
        self.assertAlmostEqual(p["tan_half_fov_y"], 12.0 / 50.0)

    def test_auto_portrait_uses_horizontal_sensor(self):
        p = MOD.camera_params(CamData(), 0.5)
        self.assertAlmostEqual(p["tan_half_fov_y"], (18.0 / 50.0) / 0.5)

    def test_shift_passthrough(self):
        p = MOD.camera_params(CamData(shift_x=0.3, shift_y=-0.1), 16 / 9)
        self.assertAlmostEqual(p["shift_x"], 0.3)
        self.assertAlmostEqual(p["shift_y"], -0.1)


class TypedLightRowTests(unittest.TestCase):
    def test_point_row_defaults_and_soft_size(self):
        row = MOD.typed_light_row(
            identity_matrix((1, 2, 3)),
            LightData(type="POINT", energy=80.0, shadow_soft_size=0.25))
        self.assertEqual(len(row), 16)
        self.assertEqual(row[3], 0.0)              # type point
        self.assertEqual(tuple(row[0:3]), (1.0, 2.0, 3.0))
        self.assertAlmostEqual(row[7], 8.0)        # energy * 0.1
        self.assertAlmostEqual(row[13], 0.25)      # soft size in ax slot

    def test_sun_direction_is_local_minus_z(self):
        # Rotate 180 deg around Z: local -Z stays -Z.
        rot = [(-1, 0, 0), (0, -1, 0), (0, 0, 1)]
        row = MOD.typed_light_row(identity_matrix((0, 0, 0), rot),
                                  LightData(type="SUN", angle=0.5))
        self.assertAlmostEqual(row[3], 1.0)
        self.assertAlmostEqual(row[4], 0.0, places=6)
        self.assertAlmostEqual(row[5], 0.0, places=6)
        self.assertAlmostEqual(row[6], -1.0, places=6)
        self.assertAlmostEqual(row[11], 0.5)       # sun angle slot

    def test_spot_cone_cosines(self):
        row = MOD.typed_light_row(identity_matrix(),
                                  LightData(type="SPOT", spot_size=math.pi / 3,
                                            spot_blend=0.0))
        self.assertAlmostEqual(row[11], math.cos(math.pi / 6))   # outer
        self.assertAlmostEqual(row[12], math.cos(math.pi / 6))   # blend=0 inner==outer

    def test_area_rectangle_extents(self):
        row = MOD.typed_light_row(identity_matrix(),
                                  LightData(type="AREA", shape="RECTANGLE",
                                            size=2.0, size_y=4.0))
        self.assertAlmostEqual(row[3], 3.0)
        self.assertAlmostEqual(row[11], 2.0)
        self.assertAlmostEqual(row[12], 4.0)
        # axis must be unit length and perpendicular to forward (-Z here).
        axv = row[13:16]
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in axv)), 1.0, places=5)
        self.assertAlmostEqual(abs(row[6]), 1.0, places=5)  # fwd.z component


def principled_material(rgb):
    """Fake material using nodes with an unlinked Base Color input."""
    base = types.SimpleNamespace(is_linked=False, default_value=(*rgb, 1.0))
    node = types.SimpleNamespace(type="BSDF_PRINCIPLED",
                                 inputs={"Base Color": base})
    mat = types.SimpleNamespace(use_nodes=True,
                                node_tree=types.SimpleNamespace(nodes=[node]),
                                diffuse_color=(0.9, 0.9, 0.9, 1.0))
    return mat


class MaterialColorTests(unittest.TestCase):
    def test_principled_base_color_squared_to_linear(self):
        mat = principled_material((1.0, 0.5, 0.25))
        r, g, b = MOD._material_rgb(mat)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(g, 0.25)
        self.assertAlmostEqual(b, 0.0625)

    def test_non_node_falls_back_to_diffuse(self):
        mat = types.SimpleNamespace(use_nodes=False,
                                    diffuse_color=(0.5, 1.0, 0.25, 1.0))
        r, g, b = MOD._material_rgb(mat)
        self.assertAlmostEqual(r, 0.25)
        self.assertAlmostEqual(g, 1.0)
        self.assertAlmostEqual(b, 0.0625)

    def test_none_material_gives_default(self):
        self.assertEqual(MOD._material_rgb(None), (0.8, 0.8, 0.82))


class WorldLightTests(unittest.TestCase):
    def _scene(self, world):
        return types.SimpleNamespace(world=world)

    def _background(self, color, strength, linked=False):
        inp_c = types.SimpleNamespace(is_linked=linked,
                                      default_value=(*color, 1.0),
                                      links=[])
        inp_s = types.SimpleNamespace(is_linked=False, default_value=strength)
        node = types.SimpleNamespace(type="BACKGROUND",
                                     inputs={"Color": inp_c, "Strength": inp_s})
        w = types.SimpleNamespace(use_nodes=True,
                                  node_tree=types.SimpleNamespace(nodes=[node]),
                                  color=(1, 1, 1))
        return w

    def test_unlinked_color_and_strength(self):
        s = self._scene(self._background((0.5, 1.0, 2.0), 3.0, linked=False))
        r, g, b, k = MOD._world_light(s)
        self.assertAlmostEqual(r, 0.25 * 1.0)
        self.assertAlmostEqual(g, 1.0)
        self.assertAlmostEqual(b, 4.0)
        self.assertAlmostEqual(k, 3.0)

    def test_linked_color_falls_back_white_but_strength_honored(self):
        s = self._scene(self._background((9, 9, 9), 2.0, linked=True))
        r, g, b, k = MOD._world_light(s)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(g, 1.0)
        self.assertAlmostEqual(b, 1.0)
        self.assertAlmostEqual(k, 2.0)

    def test_no_world_returns_none(self):
        self.assertIsNone(MOD._world_light(self._scene(None)))


class CorePathResolutionTests(unittest.TestCase):
    def test_repo_bin_resolves_exe(self):
        if not (os.path.isfile(os.path.join(REPO_ROOT, "bin", "Rstr2Core.exe"))
                or os.path.isfile(os.path.join(ADDON_DIR, "bin",
                                               "Rstr2Core.exe"))):
            self.skipTest("no core binary present (fresh checkout / CI)")
        exe = MOD.core_proc.core_exe_path()
        self.assertIsNotNone(exe, "exe must resolve from repo or addon-local bin")
        self.assertTrue(str(exe).endswith("Rstr2Core.exe"))
        norm = os.path.normcase(os.path.abspath(str(exe)))
        repo_bin = os.path.normcase(
            os.path.join(REPO_ROOT, "bin", "Rstr2Core.exe"))
        local_bin = os.path.normcase(
            os.path.join(ADDON_DIR, "bin", "Rstr2Core.exe"))
        self.assertIn(norm, {repo_bin, local_bin})

    def test_env_override_considered(self):
        import importlib
        from pathlib import Path
        cp = importlib.import_module("addon.core_proc")
        fake = Path(REPO_ROOT) / "tests"
        os.environ["RSTR2_CORE_BIN"] = str(fake)
        try:
            dirs = [str(d) for d in cp._candidate_dirs()]
            self.assertIn(str(fake), dirs)
        finally:
            del os.environ["RSTR2_CORE_BIN"]


if __name__ == "__main__":
    unittest.main()
