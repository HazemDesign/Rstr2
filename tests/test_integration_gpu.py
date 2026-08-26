# GPU integration suite: drives the REAL Rstr2Core.exe end-to-end through the
# shared-memory bridges and asserts pixel-level behavior (ReSTIR light types,
# world ambient, film alpha, dynamic resize, TAA stability).
#
# Requires: Windows + NVIDIA driver + built binaries in addon/bin or repo bin.
# Skipped automatically when RSTR2_SKIP_GPU is set (e.g. in CI).

import math
import os
import subprocess
import time
import unittest

import numpy as np

from util import load_addon, ADDON_DIR

MOD = load_addon()
S = MOD.scene_shm

CREATE_NO_WINDOW = 0x08000000
exe = MOD.core_proc.core_exe_path()

GPU_TESTS_ENABLED = (
    os.name == "nt"
    and not os.environ.get("RSTR2_SKIP_GPU")
    and exe is not None
)


def wall_scene():
    verts = np.array(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], dtype=np.float32)
    inds = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    cam = {"origin": (0.0, 0.6, 3.0), "right": (1.0, 0.0, 0.0),
           "up": (0.0, 1.0, 0.0), "forward": (0.0, 0.0, -1.0),
           "tan_half_fov_y": 0.483}
    return verts, inds, cam


def tri_scene():
    verts = np.array([[-0.7, -0.5, 0], [0.7, -0.5, 0], [0, 0.7, 0]],
                     dtype=np.float32)
    inds = np.array([0, 1, 2], dtype=np.uint32)
    cam = {"origin": (0.0, 0.6, -2.2), "right": (1.0, 0.0, 0.0),
           "up": (0.0, 1.0, 0.0), "forward": (0.0, 0.0, 1.0),
           "tan_half_fov_y": math.tan(0.45)}
    return verts, inds, cam


@unittest.skipUnless(GPU_TESTS_ENABLED,
                     "GPU integration requires Windows + Rstr2Core.exe")
class CoreIntegration(unittest.TestCase):
    """Each test runs against a FRESH core process so accumulation state can
    never leak between scenarios (matches how a user attaches to a new core)."""

    def setUp(self):
        self.proc = subprocess.Popen(
            [str(exe)], cwd=str(os.path.dirname(exe)),
            creationflags=CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.reader = MOD.shm.CoreFrameReader()
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("core exited during startup")
            if self.reader.open():
                f = self.reader.read_frame()
                if f is not None:
                    break
            time.sleep(0.25)
        else:
            self._kill()
            raise RuntimeError("core did not produce a frame in 20 s")
        self.writer = S.SceneWriter()
        if not self.writer.is_open():
            self._kill()
            raise RuntimeError("scene writer failed to open")

    def tearDown(self):
        self._kill()

    def _kill(self):
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        try:
            if self.reader is not None:
                self.reader.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _publish(self, verts, inds, cam, lights=None, albedos=None,
                 settings=None):
        self.assertTrue(self.writer.write(verts, inds, cam, lights, albedos,
                                          settings))

    def _settle(self, frames=40, timeout=8.0):
        """Wait until frame_index advanced `frames` past current (EMA settle)."""
        f = self.reader.read_frame()
        base = f[0] if f else 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            f = self.reader.read_frame()
            if f is not None and f[0] >= base + frames:
                return f[1]
            time.sleep(0.05)
        raise AssertionError("frame did not settle in %.1f s" % timeout)

    def _px(self, img, fx, fy):
        h, w = img.shape[:2]
        return img[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))][:3]

    def _alpha(self, img, fx, fy):
        h, w = img.shape[:2]
        return img[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))][3]

    def _point_lights(self):
        return np.array(
            [[-1.5, 1.0, 1.5, 0.0, 0, 0, 0, 8.0, 1, 1, 1, 0, 0, 0, 0, 0]],
            dtype=np.float32)

    # ------------------------------------------------------------------
    def test_01_point_light_falloff_monotonic(self):
        v, i, c = wall_scene()
        self._publish(v, i, c, lights=self._point_lights())
        img = self._settle()
        left = float(np.mean(self._px(img, 0.25, 0.5)))
        center = float(np.mean(self._px(img, 0.50, 0.5)))
        right = float(np.mean(self._px(img, 0.75, 0.5)))
        self.assertGreater(left, center, "falloff must decrease away from light")
        self.assertGreater(center, right, "falloff must decrease away from light")
        self.assertTrue(np.all(np.isfinite(img[:, :, :3])), "non-finite pixels")

    def test_02_sun_is_uniform_across_wall(self):
        d = np.array([0.25, -0.35, -0.90]); d /= np.linalg.norm(d)
        sun = np.array([[0, 0, 0, 1.0, d[0], d[1], d[2], 2.5,
                         1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float32)
        v, i, c = wall_scene()
        self._publish(v, i, c, lights=sun)
        img = self._settle()
        samples = [float(np.mean(self._px(img, fx, fy)))
                   for fx, fy in ((0.2, 0.3), (0.8, 0.3), (0.2, 0.7),
                                  (0.8, 0.7), (0.5, 0.5))]
        self.assertLess(max(samples) - min(samples), 0.03,
                        "sun must be uniform (no distance falloff)")

    def test_03_spot_cone_cuts_off_outside(self):
        pos = np.array([0.0, 1.5, 3.0]); aim = np.array([1.5, 0.5, 0.0])
        d = aim - pos; d /= np.linalg.norm(d)
        co, ci = np.cos(np.radians(30)), np.cos(np.radians(21))
        spot = np.array([[pos[0], pos[1], pos[2], 2.0, d[0], d[1], d[2], 40.0,
                          1, 1, 1, co, ci, 0, 0, 0]], dtype=np.float32)
        v, i, c = wall_scene()
        self._publish(v, i, c, lights=spot)
        img = self._settle()
        inside = float(np.mean(self._px(img, 0.75, 0.45)))
        outside = float(np.mean(self._px(img, 0.10, 0.15)))
        self.assertGreater(inside, 0.3, "cone interior must be lit")
        self.assertLess(outside, 0.01, "outside cone must be black")

    def test_04_world_ambient_tints_and_scales(self):
        v, i, c = wall_scene()
        base = {"flags": 1, "exposure": 1.0, "history": 8}
        blue = [0.35, 0.5, 1.0]
        self._publish(v, i, c, None, None,
                      dict(base, world=[*blue, 0.5]))
        weak = self._settle()
        self._publish(v, i, c, None, None,
                      dict(base, world=[*blue, 4.0]))
        strong = self._settle()
        for img in (weak, strong):
            m = img[:, :, :3].mean(axis=(0, 1))
            self.assertGreater(m[2], m[0] + 0.05, "blue world must tint scene")
        self.assertGreater(float(strong[:, :, 2].mean()),
                           float(weak[:, :, 2].mean()) * 1.5,
                           "world strength must scale ambient")

    def test_05_film_transparent_alpha(self):
        v, i, c = tri_scene()   # small triangle -> background visible
        self._publish(v, i, c, None, None,
                      {"flags": 3, "history": 8,
                       "world": [0.35, 0.5, 1.0, 1.2]})
        img = self._settle()
        self.assertLess(self._alpha(img, 0.02, 0.02), 1e-6,
                        "background alpha must be 0 with film_transparent")
        self.assertAlmostEqual(self._alpha(img, 0.5, 0.75), 1.0, places=5,
                               msg="geometry alpha must stay 1")

    def test_06_dynamic_resize_honored(self):
        v, i, c = wall_scene()
        self._publish(v, i, c, None, None,
                      {"flags": 1, "size": [800, 600]})
        deadline = time.time() + 6
        got = None
        while time.time() < deadline:
            f = self.reader.read_frame()
            if f is not None and f[1].shape[1] == 800 and f[1].shape[0] == 600:
                got = f[1]
                break
            time.sleep(0.05)
        self.assertIsNotNone(got, "core did not honor requested 800x600")

    def test_07_taa_output_stable_when_static(self):
        v, i, c = wall_scene()
        self._publish(v, i, c, lights=self._point_lights(),
                      settings={"flags": 1, "history": 20})
        a = self._settle(frames=60)
        b = self._settle(frames=30)
        diff = abs(float(a[:, :, :3].mean()) - float(b[:, :, :3].mean()))
        self.assertLess(diff, 2e-3, "static scene must converge to stable mean")


if __name__ == "__main__":
    unittest.main()
