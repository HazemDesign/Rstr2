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
    "version": (0, 3, 0),
    "blender": (4, 0, 0),
    "location": "Render Properties > Render Engine > Rstr2",
    "description": "Custom RT renderer (Phase 3: DXR scene sync from Blender)",
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


def _extract_scene(depsgraph):
    """Build (vertices, indices, camera) for the native core.

    vertices : (n, 3) float32, world-space triangle-soup vertices
    indices  : (m,) uint32, 3 per triangle, indexing into `vertices`
    camera   : dict with origin/right/up/forward (each length-3) and
               tan_half_fov_y (float)

    Returns None when the scene has no mesh geometry (so the core keeps its
    last/ default scene instead of failing on an empty update).
    """
    import math

    # --- Camera basis from the active camera (or a sensible default) -----
    camera = depsgraph.scene.camera
    if camera is None:
        cam = {
            "origin": (0.0, 0.6, -2.2),
            "right": (1.0, 0.0, 0.0),
            "up": (0.0, 1.0, 0.0),
            "forward": (0.0, 0.0, 1.0),
            "tan_half_fov_y": math.tan(0.45),
        }
    else:
        cam_eval = camera.evaluated_get(depsgraph)
        mw = cam_eval.matrix_world
        origin = (float(mw[0][3]), float(mw[1][3]), float(mw[2][3]))
        right = (float(mw[0][0]), float(mw[1][0]), float(mw[2][0]))
        up = (float(mw[0][1]), float(mw[1][1]), float(mw[2][1]))
        back = (float(mw[0][2]), float(mw[1][2]), float(mw[2][2]))
        forward = (-back[0], -back[1], -back[2])
        cd = camera.data
        fov_y = getattr(cd, "angle_y", None)
        if fov_y is None:
            fov_y = cd.angle
        cam = {
            "origin": origin,
            "right": right,
            "up": up,
            "forward": forward,
            "tan_half_fov_y": math.tan(float(fov_y) / 2.0),
        }

    # --- World-space triangle soup from all MESH objects -----------------
    vert_chunks = []
    idx_chunks = []
    offset = 0
    for obj in depsgraph.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.evaluated_get(depsgraph).data
        n = len(mesh.vertices)
        if n == 0:
            continue

        local = np.empty((n, 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", local.ravel())

        m = np.array(obj.matrix_world, dtype=np.float32)
        ones = np.ones((n, 1), dtype=np.float32)
        vh = np.concatenate([local, ones], axis=1)  # (n, 4)
        world = (m @ vh.T).T[:, :3].astype(np.float32)
        vert_chunks.append(world)

        loops = mesh.loops
        li = []
        for tri in mesh.loop_triangles:
            for lp in tri.loops:
                li.append(offset + int(loops[lp].vertex_index))
        idx_chunks.append(np.asarray(li, dtype=np.uint32))
        offset += n

    if not vert_chunks:
        return None

    vertices = np.concatenate(vert_chunks, axis=0)
    indices = np.concatenate(idx_chunks, axis=0) if idx_chunks else np.empty((0,), dtype=np.uint32)

    # --- Point lights (world space) --------------------------------------
    light_rows = []
    for obj in depsgraph.objects:
        if obj.type != "LIGHT":
            continue
        ld = obj.data
        mw = obj.matrix_world
        pos = (float(mw[0][3]), float(mw[1][3]), float(mw[2][3]))
        energy = float(getattr(ld, "energy", 10.0))
        col = ld.color
        cr, cg, cb = float(col[0]), float(col[1]), float(col[2])
        intensity = energy * 0.1  # scale Blender energy into our radiance range
        light_rows.append([pos[0], pos[1], pos[2], intensity, cr, cg, cb, 0.0])
    lights = (np.array(light_rows, dtype=np.float32).reshape(-1, 8)
              if light_rows else np.empty((0, 8), dtype=np.float32))

    return vertices, indices, cam, lights


class Rstr2Engine(RenderEngine):
    bl_idname = "RSTR2"
    bl_label = "Rstr2"
    bl_use_postprocess = True

    def __init__(self, *args, **kwargs):
        # Blender instantiates RenderEngine subclasses with an extra argument
        # (the engine type); forward it to the base class so __init__ never
        # throws during engine creation.
        super().__init__(*args, **kwargs)
        self.frame = 0

        # --- Phase 2/3: native core bridge state -----------------------
        self._core_started = False
        self._core_ok = False
        self._core_msg = ""
        self._core = None          # CoreProcess instance
        self._reader = None        # CoreFrameReader instance
        self._scene_writer = None  # SceneWriter instance (addon -> core)
        self._last_frame_index = None
        self._cached_pixels = None
        self._dbg = ""             # short status shown in the viewport

        # Phase 3: camera/geometry liveness tracking.
        self._geom_dirty = True
        self._last_cam_sig = None

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
            self._sync_scene(depsgraph)
            if self._ensure_core():
                frame = self._reader.read_frame()
                if frame is not None:
                    _, pf = frame
                    pixels = _resize_nearest(pf, width, height)
        except Exception as e:
            self._dbg = "render err: %s" % str(e)[:60]
            pixels = None

        if pixels is None:
            self.update_stats("", "Rstr2: rendering test pattern")
            pixels = _make_test_pattern(width, height)
        else:
            self.update_stats("", "Rstr2: rendering live core")

        # Buffer row 0 is the TOP of the image; Blender's result rect is
        # bottom-up, so flip rows to display upright.
        pixels = pixels[::-1, :, :]

        result = self.begin_result(0, 0, width, height)
        layer = result.layers[0].passes["Combined"]
        layer.rect = pixels.reshape(-1, 4)
        self.end_result(result)
        self.update_stats("", "Rstr2: done")

    # ------------------------------------------------------------------
    # Viewport sync
    # ------------------------------------------------------------------
    def view_update(self, context, depsgraph):
        # Phase 3: flag the scene (meshes + camera) for re-publish to the
        # native core. The actual publish happens in view_draw so it runs on
        # every redraw and stays live during camera orbit/zoom.
        self._geom_dirty = True
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

        # Phase 3: keep the native core fed with the current camera + scene so
        # orbit/zoom in the viewport reach the renderer live. We only re-publish
        # when the camera actually moved (or geometry was flagged dirty), which
        # the SceneWriter then coalesces to avoid needless BVH rebuilds.
        try:
            cam_sig = self._camera_sig(depsgraph)
            if self._geom_dirty or cam_sig != self._last_cam_sig:
                self._sync_scene(depsgraph)
                self._last_cam_sig = cam_sig
                self._geom_dirty = False
        except Exception as e:
            self._dbg = "sync err: %s" % str(e)[:60]

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
        except Exception as e:
            self._dbg = "draw err: %s" % str(e)[:60]
            live = False

        if live:
            status = "Rstr2: live core | " + self._dbg
        else:
            pixels = _make_test_pattern(width, height)
            status = "Rstr2: waiting | " + self._core_msg + " | " + self._dbg

        self.update_stats("", status)

        texture = self._upload_texture(pixels)

        shader = gpu.shader.from_builtin("IMAGE")
        # Blender 5.x removed the `gpu.batch` helper; build the batch from
        # gpu.types directly so this works across versions.
        from gpu.types import GPUVertBuf, GPUVertFormat, GPUBatch

        vformat = GPUVertFormat()
        pos_id = vformat.attr_add(id="pos", comp_type="F32", len=2, fetch_mode="FLOAT")
        uv_id = vformat.attr_add(id="texCoord", comp_type="F32", len=2, fetch_mode="FLOAT")
        vbo = GPUVertBuf(vformat, 4)
        vbo.attr_fill(id=pos_id, data=[(0, 0), (width, 0), (width, height), (0, height)])
        # Flip V so the shared-memory buffer (first row = TOP) maps upright:
        # screen-top samples buffer row 0 instead of the last row.
        vbo.attr_fill(id=uv_id, data=[(0, 1), (1, 1), (1, 0), (0, 0)])
        batch = GPUBatch(type="TRI_FAN", buf=vbo)

        gpu.state.blend_set("ALPHA_PREMULT")
        shader.uniform_sampler("image", texture)
        batch.draw(shader)
        gpu.state.blend_set("NONE")

        # Keep redrawing while the engine is active so the live feed updates.
        self.tag_redraw()

        # TODO(Phase 2): wrap this in bind_display_space_shader(scene)/unbind
        # once we display real linear output from the native core.


    # ------------------------------------------------------------------
    # Phase 3 scene bridge helpers
    # ------------------------------------------------------------------
    def _camera_sig(self, depsgraph):
        """Cheap signature of the active camera (basis + vertical FOV).

        Used by view_draw to re-publish the scene only when the camera
        actually moved, so orbit/zoom in the viewport reaches the core live.
        """
        cam = depsgraph.scene.camera
        if cam is None:
            return ("default",)
        cd = cam.data
        fov_y = getattr(cd, "angle_y", None)
        if fov_y is None:
            fov_y = cd.angle
        mw = cam.matrix_world
        return (
            round(float(mw[0][3]), 4), round(float(mw[1][3]), 4), round(float(mw[2][3]), 4),
            round(float(mw[0][0]), 4), round(float(mw[1][0]), 4), round(float(mw[2][0]), 4),
            round(float(mw[0][1]), 4), round(float(mw[1][1]), 4), round(float(mw[2][1]), 4),
            round(float(mw[0][2]), 4), round(float(mw[1][2]), 4), round(float(mw[2][2]), 4),
            round(float(fov_y), 4),
        )

    def _ensure_scene_writer(self):
        """Lazily create the addon->core scene mapping writer."""
        if self._scene_writer is None:
            try:
                from . import scene_shm

                self._scene_writer = scene_shm.SceneWriter()
            except Exception:
                self._scene_writer = None
        return self._scene_writer is not None

    def _sync_scene(self, depsgraph):
        """Extract meshes + camera from the depsgraph and publish to the core."""
        scene = _extract_scene(depsgraph)
        if scene is None:
            self._dbg = "no mesh"
            return
        if not self._ensure_scene_writer():
            self._dbg = "scene map n/a"
            return
        try:
            vertices, indices, camera, lights = scene
            ok = self._scene_writer.write(vertices, indices, camera, lights)
            self._dbg = ("scene %d v / %d i / %d L" % (vertices.shape[0], indices.shape[0], lights.shape[0])) if ok else "scene write fail"
        except Exception as e:
            self._dbg = "scene err: %s" % str(e)[:60]

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
                ptx = core_proc.core_ptx_path()
                if exe is None:
                    self._core_ok = False
                    self._core_msg = "Rstr2Core.exe missing in bin/"
                elif ptx is None:
                    self._core_ok = False
                    self._core_msg = "optix_kernels.ptx missing in bin/"
                else:
                    self._core = core_proc.CoreProcess()
                    self._core.launch()
                    self._core_ok = True
                    self._core_msg = "core launched"
            except Exception as e:
                self._core_ok = False
                self._core_msg = "core launch err: %s" % str(e)[:60]

        if not self._core_ok:
            return False

        # Open the mapping lazily (the core may need a few frames to create it).
        if self._reader is None:
            try:
                from . import shm

                reader = shm.CoreFrameReader()
                if reader.open():
                    self._reader = reader
                    self._core_msg = "frame map open"
                else:
                    code = self._core.exit_code() if self._core else None
                    self._core_msg = "frame map not open (core exited, code %s)" % str(code)
            except Exception as e:
                self._core_msg = "frame open err: %s" % str(e)[:40]

        return self._reader is not None

    def _shutdown_core(self):
        """Terminate the core process and close the mappings (best effort)."""
        try:
            if self._reader is not None:
                self._reader.close()
                self._reader = None
        except Exception:
            pass
        try:
            if self._scene_writer is not None:
                self._scene_writer.close()
                self._scene_writer = None
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
