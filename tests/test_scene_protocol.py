# Unit tests for the scene shared-memory protocol (writer side).
#
# These run WITHOUT Blender or a GPU: they exercise the byte-level contract
# between the Python writer and the C++ reader (shared_mem.h v2), including
# the reserved-area addendum (world light, render size, camera shift).

import ctypes
import struct
import unittest

import numpy as np

from util import load_addon

MOD = load_addon()
S = MOD.scene_shm


def read_view_bytes(addr, off, n):
    return ctypes.string_at(addr + off, n)


class SceneProtocolTests(unittest.TestCase):
    def setUp(self):
        self.w = S.SceneWriter()
        self.assertTrue(self.w.is_open(), "SceneWriter failed to open mapping")

    def tearDown(self):
        self.w.close()

    # ------------------------------------------------------------------
    def _wall(self, lights=None, albedos=None, settings=None):
        verts = np.array(
            [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], dtype=np.float32
        )
        inds = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        cam = {
            "origin": (0.0, 0.6, 3.0),
            "right": (1.0, 0.0, 0.0),
            "up": (0.0, 1.0, 0.0),
            "forward": (0.0, 0.0, -1.0),
            "tan_half_fov_y": 0.483,
            "shift_x": 0.05,
            "shift_y": -0.02,
        }
        ok = self.w.write(verts, inds, cam, lights, albedos, settings)
        self.assertTrue(ok, "write() returned False")
        return verts, inds, cam

    def _header(self):
        raw = read_view_bytes(self.w._addr, 0, S.HEADER_SIZE)
        magic, version, epoch, ready, writing, vc, ic, lc, flags = struct.unpack_from(
            "<9I", raw, 0)
        exposure, history = struct.unpack_from("<2f", raw, 36)
        world = struct.unpack_from("<4f", raw, 96)
        size = struct.unpack_from("<2I", raw, 112)
        shift = struct.unpack_from("<2f", raw, 120)
        return dict(magic=magic, version=version, epoch=epoch, ready=ready,
                    writing=writing, vc=vc, ic=ic, lc=lc, flags=flags,
                    exposure=exposure, history=history, world=world,
                    size=size, shift=shift)

    # ------------------------------------------------------------------
    def test_header_fields_v2(self):
        settings = {"flags": 3, "exposure": 1.5, "history": 12,
                    "world": [0.1, 0.2, 0.3, 4.0], "size": [800, 600]}
        verts, inds, cam = self._wall(settings=settings)
        h = self._header()
        self.assertEqual(h["magic"], S.MAGIC)
        self.assertEqual(h["version"], 2)
        self.assertEqual(h["ready"], 1)
        self.assertEqual(h["writing"], 0)
        self.assertEqual(h["vc"], 4)
        self.assertEqual(h["ic"], 6)
        self.assertEqual(h["flags"], 3)
        self.assertAlmostEqual(h["exposure"], 1.5, places=5)
        self.assertAlmostEqual(h["history"], 12.0, places=4)
        np.testing.assert_allclose(h["world"], [0.1, 0.2, 0.3, 4.0], atol=1e-6)
        self.assertEqual(h["size"], (800, 600))
        self.assertAlmostEqual(h["shift"][0], 0.05, places=6)
        self.assertAlmostEqual(h["shift"][1], -0.02, places=6)

    def test_payload_offsets_verts_indices_lights_albedos(self):
        lights = np.array(
            [[-1.5, 1, 1.5, 0.0, 0, 0, 0, 8.0, 1, 1, 1, 0, 0, 0.25, 0, 0]],
            dtype=np.float32,
        )
        albs = np.full((4, 3), 0.5, dtype=np.float32)
        verts, inds, _ = self._wall(lights=lights, albedos=albs)

        base = S.HEADER_SIZE
        vraw = np.frombuffer(read_view_bytes(self.w._addr, base, 4 * 12),
                             dtype=np.float32)
        np.testing.assert_allclose(vraw.reshape(4, 3), verts, atol=1e-6)

        iraw = np.frombuffer(read_view_bytes(self.w._addr, base + 48, 24),
                             dtype=np.uint32)
        np.testing.assert_array_equal(iraw, inds)

        lraw = np.frombuffer(read_view_bytes(self.w._addr, base + 72, 64),
                             dtype=np.float32).reshape(1, 16)
        self.assertEqual(lraw[0][3], 0.0)      # type point
        self.assertAlmostEqual(lraw[0][7], 8.0)  # intensity
        self.assertAlmostEqual(lraw[0][13], 0.25)  # soft size in ax slot

        araw = np.frombuffer(read_view_bytes(self.w._addr, base + 136, 48),
                             dtype=np.float32).reshape(4, 3)
        np.testing.assert_allclose(araw, albs, atol=1e-6)

    def test_albedo_block_always_present_with_default(self):
        """Core reads albedos unconditionally; missing rows must become the
        kernel default gray so surfaces never render black."""
        self._wall(albedos=None)
        base = S.HEADER_SIZE + 4 * 12 + 6 * 4  # no lights in this scene
        araw = np.frombuffer(read_view_bytes(self.w._addr, base, 4 * 12),
                             dtype=np.float32).reshape(4, 3)
        np.testing.assert_allclose(
            araw, np.full((4, 3), (0.8, 0.8, 0.82)), atol=1e-6)

    def test_wrong_length_albedos_replaced_by_default(self):
        bad = np.zeros((2, 3), dtype=np.float32)
        self._wall(albedos=bad)
        base = S.HEADER_SIZE + 4 * 12 + 6 * 4
        araw = np.frombuffer(read_view_bytes(self.w._addr, base, 4 * 12),
                             dtype=np.float32).reshape(4, 3)
        np.testing.assert_allclose(
            araw, np.full((4, 3), (0.8, 0.8, 0.82)), atol=1e-6)

    def test_epoch_dedup_and_bump(self):
        self._wall()                       # first write establishes baseline
        e0 = self._header()["epoch"]
        self._wall()                       # identical -> no bump
        self.assertEqual(self._header()["epoch"], e0)
        self._wall(settings={"exposure": 2.0})  # change -> bump
        self.assertEqual(self._header()["epoch"], e0 + 1)

    def test_rejects_empty_geometry(self):
        empty = np.empty((0, 3), dtype=np.float32)
        cam = {"origin": (0, 0, 0), "right": (1, 0, 0), "up": (0, 1, 0),
               "forward": (0, 0, -1), "tan_half_fov_y": 0.4}
        ok = self.w.write(empty, np.empty(0, dtype=np.uint32), cam)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
