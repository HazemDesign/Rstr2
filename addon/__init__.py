# Rstr2 - Phase 1 skeleton
#
# A custom viewport render engine for Blender 4.x/5.x.
# Phase 1 proves the registration + framebuffer path that killed blRstr:
#   - registers engine "Rstr2" via bpy.types.RenderEngine
#   - view_draw(): CPU numpy buffer -> GPUTexture -> fullscreen quad (builtin IMAGE shader)
#   - render():    F12 final render path via begin_result/end_result
#
# No bgl, no gpu_extras.draw_texture_2d - safe on Blender 4.0+ / 5.2.

bl_info = {
    "name": "Rstr2",
    "author": "you",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Render Properties > Render Engine > Rstr2",
    "description": "Custom RT renderer skeleton (Phase 1: test pattern)",
    "category": "Render",
}

import time

import bpy
import numpy as np
from bpy.types import RenderEngine


def _make_test_pattern(width, height):
    """Animated RGBA float32 test pattern (linear values, premultiplied alpha).

    Diagonal gradient + a moving color sweep so it is obvious the engine is alive.
    """
    t = time.time()

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    u = xs / max(width - 1, 1)
    v = ys / max(height - 1, 1)

    r = u * (0.6 + 0.4 * np.sin(t * 0.9))
    g = v * (0.6 + 0.4 * np.cos(t * 1.3))
    b = ((u + v) * 0.5) * (0.6 + 0.4 * np.sin(t * 0.7 + 2.0))

    # moving diagonal band
    sweep = np.abs(((u + v) * 0.5 + t * 0.15) % 1.0 - 0.5)
    band = (sweep < 0.04).astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1) + band[..., None] * 0.8
    rgba = np.concatenate([rgb, np.ones((height, width, 1), np.float32)], axis=-1)
    return np.clip(rgba, 0.0, 1.0).astype(np.float32)


def _resize_nearest(arr, width, height):
    """Nearest-neighbour resize of an (h, w, 4) float32 array (no deps)."""
    ah, aw = arr.shape[0], arr.shape[1]
    if ah == height and aw == width:
        return arr
    ys = (np.arange(height) * ah / height).astype(np.int64)
    xs = (np.arange(width) * aw / width).astype(np.int64)
    return arr[np.ix_(ys, xs)]


class Rstr2Engine(RenderEngine):
    bl_idname = "RSTR2"
    bl_label = "Rstr2"
    bl_use_postprocess = True

    def __init__(self):
        self.frame = 0

        # --- Phase 2: native core bridge state -------------------------
        self._core_started = False
        self._core_ok = False
        self._core = None          # CoreProcess instance
        self._reader = None        # CoreFrameReader instance
        self._last_frame_index = None
        self._cached_pixels = None

    # ------------------------------------------------------------------
    # Final render (F12 / CLI)
    # ------------------------------------------------------------------
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)

        # Try a ready core frame; fall back to the test pattern unchanged.
        pixels = None
        try:
            if self._ensure_core():
                frame = self._reader.read_frame()
                if frame is not None:
                    _, pf = frame
                    pixels = _resize_nearest(pf, width, height)
        except Exception:
            pixels = None

        if pixels is None:
            self.update_stats("", "Rstr2: rendering test pattern")
            pixels = _make_test_pattern(width, height)
        else:
            self.update_stats("", "Rstr2: rendering live core")

        result = self.begin_result(0, 0, width, height)
        layer = result.layers[0].passes["Combined"]
        layer.rect = pixels.reshape(-1, 4)
        self.end_result(result)
        self.update_stats("", "Rstr2: done")

    # ------------------------------------------------------------------
    # Viewport sync
    # ------------------------------------------------------------------
    def view_update(self, context, depsgraph):
        # Phase 1: nothing to sync yet; just keep redrawing the animated pattern.
        self.tag_redraw()

    # ------------------------------------------------------------------
    # Viewport draw
    # ------------------------------------------------------------------
    def view_draw(self, context, depsgraph):
        import gpu
        from gpu.types import GPUTexture

        region = context.region
        width = max(region.width, 2)
        height = max(region.height, 2)

        # Try to pull a live core frame; keep the animated test pattern as
        # a fallback when the core is absent or mid-update.
        pixels = None
        live = False
        try:
            if self._ensure_core():
                frame = self._reader.read_frame()
                if frame is not None:
                    idx, pf = frame
                    if idx != self._last_frame_index:
                        self._last_frame_index = idx
                        self._cached_pixels = pf
                    pixels = self._cached_pixels
                    live = pixels is not None
        except Exception:
            live = False

        if live:
            status = "Rstr2: live core"
        else:
            pixels = _make_test_pattern(width, height)
            status = "Rstr2: waiting (test pattern)"

        self.update_stats("", status)

        texture = self._upload_texture(pixels)

        shader = gpu.shader.from_builtin("IMAGE")
        batch = gpu.batch.batch_for_shader(
            shader,
            "TRI_FAN",
            {
                "pos": [(0, 0), (width, 0), (width, height), (0, height)],
                "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
        )

        gpu.state.blend_set("ALPHA_PREMULT")
        shader.uniform_sampler("image", texture)
        batch.draw(shader)
        gpu.state.blend_set("NONE")

        # Keep redrawing while the engine is active so the live feed updates.
        self.tag_redraw()

        # TODO(Phase 2): wrap this in bind_display_space_shader(scene)/unbind
        # once we display real linear output from the native core.


    # ------------------------------------------------------------------
    # Phase 2 core bridge helpers
    # ------------------------------------------------------------------
    def _ensure_core(self):
        """Lazily launch the native core and open its frame mapping.

        Returns True only when a readable frame mapping is available.
        Every failure is swallowed so the test pattern keeps drawing.
        """
        # Launch the process at most once.
        if not self._core_started:
            self._core_started = True
            try:
                from . import core_proc

                exe = core_proc.core_exe_path()
                if exe is None:
                    self._core_ok = False
                else:
                    self._core = core_proc.CoreProcess()
                    self._core.launch()
                    self._core_ok = True
            except Exception:
                self._core_ok = False

        if not self._core_ok:
            return False

        # Open the mapping lazily (the core may need a few frames to create it).
        if self._reader is None:
            try:
                from . import shm

                reader = shm.CoreFrameReader()
                if reader.open():
                    self._reader = reader
            except Exception:
                pass

        return self._reader is not None

    def _shutdown_core(self):
        """Terminate the core process and close the mapping (best effort)."""
        try:
            if self._reader is not None:
                self._reader.close()
                self._reader = None
        except Exception:
            pass
        try:
            if self._core is not None:
                self._core.terminate()
                self._core = None
        except Exception:
            pass
        self._core_ok = False
        self._core_started = False

    def _upload_texture(self, pixels):
        import gpu

        h, w = pixels.shape[0], pixels.shape[1]
        buf = gpu.types.Buffer("FLOAT", pixels.size)
        try:
            buf.from_numpy(pixels.ravel())
        except AttributeError:
            buf = gpu.types.Buffer("FLOAT", pixels.size, pixels.ravel().tolist())
        return gpu.types.GPUTexture((w, h), format="RGBA32F", data=buf)

    def __del__(self):
        try:
            self._shutdown_core()
        except Exception:
            pass


def register():
    bpy.utils.register_class(Rstr2Engine)


def unregister():
    bpy.utils.unregister_class(Rstr2Engine)


if __name__ == "__main__":
    register()
