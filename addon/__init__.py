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
    "version": (0, 5, 0),
    "blender": (4, 0, 0),
    "location": "Render Properties > Render Engine > Rstr2",
    "description": "Custom RT renderer (OptiX + ReSTIR DI: typed lights, "
                   "per-material albedo, TAA)",
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


def camera_params(cd, target_aspect):
    """Pure camera math from a Blender Camera data-block.

    Returns dict(tan_half_fov_y, shift_x, shift_y) honoring lens, sensor
    fit/size and frame offsets for the given render-target aspect (w/h)."""
    import math

    lens = max(float(getattr(cd, "lens", 50.0)), 1e-4)
    fit = getattr(cd, "sensor_fit", "AUTO")
    sw = float(getattr(cd, "sensor_width", 36.0))
    sh = float(getattr(cd, "sensor_height", 24.0))
    if fit == "VERTICAL":
        tan_v = (sh * 0.5) / lens
    elif fit == "HORIZONTAL":
        tan_h = (sw * 0.5) / lens
        tan_v = tan_h / max(target_aspect, 1e-4)
    else:  # AUTO: vertical fit unless the target is taller than wide
        if target_aspect >= 1.0:
            tan_v = (sh * 0.5) / lens
        else:
            tan_v = ((sw * 0.5) / lens) / max(target_aspect, 1e-4)
    return {
        "tan_half_fov_y": tan_v,
        "shift_x": float(getattr(cd, "shift_x", 0.0)),
        "shift_y": float(getattr(cd, "shift_y", 0.0)),
    }


def typed_light_row(mw, ld):
    """Build one 16-float typed-light row from a light object's world matrix
    and data-block (see scene_shm LIGHT_FLOATS for the layout). Pure/untested
    against Blender itself - unit tests cover the math."""
    import math

    pos = (float(mw[0][3]), float(mw[1][3]), float(mw[2][3]))
    energy = float(getattr(ld, "energy", 10.0))
    col = ld.color
    cr, cg, cb = float(col[0]), float(col[1]), float(col[2])
    # Emission direction = local -Z of the light object (Blender convention).
    fwd = (-float(mw[0][2]), -float(mw[1][2]), -float(mw[2][2]))
    flen = math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2)
    if flen > 1e-9:
        fwd = (fwd[0] / flen, fwd[1] / flen, fwd[2] / flen)
    else:
        fwd = (0.0, -1.0, 0.0)
    ltype = getattr(ld, "type", "POINT")

    if ltype == "SUN":
        intensity = energy * 3.0  # sun strength ~ irradiance; brighten a touch
        return [pos[0], pos[1], pos[2], 1.0,
                fwd[0], fwd[1], fwd[2], intensity,
                cr, cg, cb, float(getattr(ld, "angle", 0.0)), 0.0,
                0.0, 0.0, 0.0]
    if ltype == "SPOT":
        spot_size = float(getattr(ld, "spot_size", 0.8))
        blend = float(getattr(ld, "spot_blend", 0.15))
        cos_outer = math.cos(spot_size * 0.5)
        cos_inner = math.cos(spot_size * 0.5 * max(1.0 - blend, 0.001))
        intensity = energy * 0.1
        # az carries shadow_soft_size (spherical emitter radius).
        return [pos[0], pos[1], pos[2], 2.0,
                fwd[0], fwd[1], fwd[2], intensity,
                cr, cg, cb, cos_outer, cos_inner,
                0.0, 0.0, float(getattr(ld, "shadow_soft_size", 0.0))]
    if ltype == "AREA":
        size_x = float(getattr(ld, "size", 1.0))
        shape = getattr(ld, "shape", "SQUARE")
        size_y = float(getattr(ld, "size_y", size_x)) if shape == "RECTANGLE" else size_x
        ref = (0.0, 1.0, 0.0) if abs(fwd[1]) < 0.9 else (1.0, 0.0, 0.0)
        ax = (ref[1] * fwd[2] - ref[2] * fwd[1],
              ref[2] * fwd[0] - ref[0] * fwd[2],
              ref[0] * fwd[1] - ref[1] * fwd[0])
        alen = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)
        if alen > 1e-9:
            ax = (ax[0] / alen, ax[1] / alen, ax[2] / alen)
        else:
            ax = (1.0, 0.0, 0.0)
        intensity = energy * 0.1
        return [pos[0], pos[1], pos[2], 3.0,
                fwd[0], fwd[1], fwd[2], intensity,
                cr, cg, cb, size_x, size_y,
                ax[0], ax[1], ax[2]]
    # POINT (default)
    intensity = energy * 0.1  # scale Blender energy into our radiance range
    # ax carries shadow_soft_size (spherical emitter radius).
    return [pos[0], pos[1], pos[2], 0.0,
            0.0, 0.0, 0.0, intensity,
            cr, cg, cb, 0.0, 0.0,
            float(getattr(ld, "shadow_soft_size", 0.0)), 0.0, 0.0]


def _extract_scene(depsgraph, view_override=None, target_aspect=16.0 / 9.0):
    """Build (vertices, indices, camera, lights, albedos) for the native core.

    vertices : (n, 3) float32, world-space triangle-soup vertices
    indices  : (m,) uint32, 3 per triangle, indexing into `vertices`
    camera   : dict with origin/right/up/forward (each length-3) and
               tan_half_fov_y (float)
    lights   : (L, 16) float32 typed-light rows
               [px,py,pz,type,dx,dy,dz,intensity,cr,cg,cb,sx,sy,ax,ay,az]
               types: 0 point, 1 sun, 2 spot, 3 area
    albedos  : (n, 3) float32 linear RGB per vertex (from material base color)

    view_override: optional dict with origin/right/up/forward/tan_half_fov_y
    describing the actual 3D viewport viewpoint (orbit/pan/zoom), used instead
    of the scene camera so free navigation renders what you see.
    target_aspect: w/h of the render target, for horizontal-sensor-fit cameras.

    Returns None when the scene has no mesh geometry (so the core keeps its
    last/ default scene instead of failing on an empty update).
    """
    import math

    # --- Camera basis: viewport override > active camera > default --------
    camera = depsgraph.scene.camera
    if view_override is not None:
        cam = dict(view_override)
    elif camera is None:
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
        cam = dict(
            origin=origin, right=right, up=up, forward=forward,
            **camera_params(cd, target_aspect)
        )

    # --- World-space triangle soup from all MESH objects -----------------
    # Non-indexed soup: one vertex per triangle corner, so each triangle owns
    # its three verts exclusively. This lets albedo be exact per triangle even
    # when meshes share vertices across different materials.
    vert_chunks = []
    idx_chunks = []
    alb_chunks = []
    offset = 0
    for obj in depsgraph.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.evaluated_get(depsgraph).data
        nt = len(mesh.loop_triangles)
        if nt == 0:
            continue

        n = len(mesh.vertices)
        local = np.empty((n, 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", local.ravel())

        # Corner -> source-vertex map for every loop_triangle.
        nt_loops = len(mesh.loops)
        loops_vertex = np.empty(nt_loops, dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loops_vertex)
        corner_l = np.empty(nt * 3, dtype=np.int64)
        w = 0
        for tri in mesh.loop_triangles:
            for lp in tri.loops:
                corner_l[w] = lp
                w += 1
        corner_v = loops_vertex[corner_l]
        m = np.array(obj.matrix_world, dtype=np.float32)
        vh = np.concatenate([local[corner_v], np.ones((nt * 3, 1), dtype=np.float32)], axis=1)
        world = (m @ vh.T).T[:, :3].astype(np.float32)

        materials = list(getattr(mesh, "materials", None) or [])
        mat_rgb = np.empty((nt, 3), dtype=np.float32)
        for t, tri in enumerate(mesh.loop_triangles):
            rgb = (0.8, 0.8, 0.82)
            mi = int(tri.material_index)
            if 0 <= mi < len(materials):
                rgb = _material_rgb(materials[mi])
            mat_rgb[t] = rgb

        vert_chunks.append(world)
        idx_chunks.append(np.arange(nt * 3, dtype=np.uint32) + offset)
        alb_chunks.append(np.repeat(mat_rgb, 3, axis=0).astype(np.float32))
        offset += nt * 3

    if not vert_chunks:
        return None

    vertices = np.concatenate(vert_chunks, axis=0)
    indices = np.concatenate(idx_chunks, axis=0) if idx_chunks else np.empty((0,), dtype=np.uint32)
    albedos = (np.concatenate(alb_chunks, axis=0)
               if alb_chunks else np.empty((0, 3), dtype=np.float32))

    # --- Typed lights (point / sun / spot / area), world space ------------
    light_rows = []
    for obj in depsgraph.objects:
        if obj.type != "LIGHT":
            continue
        light_rows.append(typed_light_row(obj.matrix_world, obj.data))

    lights = (np.array(light_rows, dtype=np.float32).reshape(-1, 16)
              if light_rows else np.empty((0, 16), dtype=np.float32))

    return vertices, indices, cam, lights, albedos


def _world_light(scene):
    """Uniform world/environment radiance (r, g, b, strength) or None.

    Reads the World's Background node (the common case). Color is converted
    from sRGB with the same gamma-2.0 approximation used for materials.
    If the Color input is linked (env textures etc.) we fall back to white
    times Strength so strength edits still respond live."""
    try:
        world = scene.world
        if world is None:
            return None
        strength = 1.0
        color = (1.0, 1.0, 1.0)
        if getattr(world, "use_nodes", False) and world.node_tree:
            for node in world.node_tree.nodes:
                if node.type == "BACKGROUND":
                    inp_c = node.inputs.get("Color")
                    inp_s = node.inputs.get("Strength")
                    if inp_c is not None:
                        if not inp_c.is_linked:
                            c = inp_c.default_value
                            color = (float(c[0]), float(c[1]), float(c[2]))
                        elif getattr(inp_c, "links", None):
                            try:
                                src = inp_c.links[0].from_node
                                if src.type == "RGB":
                                    c = src.outputs[0].default_value
                                    color = (float(c[0]), float(c[1]),
                                             float(c[2]))
                                # env/sky textures: keep white; only the
                                # uniform approximation exists for now.
                            except Exception:
                                pass
                    if inp_s is not None and not inp_s.is_linked:
                        strength = float(inp_s.default_value)
                    break
        else:
            c = world.color
            color = (float(c[0]), float(c[1]), float(c[2]))
        lin = tuple(x * x for x in color)
        return (lin[0], lin[1], lin[2], strength)
    except Exception:
        return None


def _material_rgb(mat):
    """Linear base-color RGB for a material (Principled Base Color preferred).

    Note: Blender colors are sRGB-encoded; converting exactly would need the
    scene view transform. We approximate with the common gamma-2.0 shortcut,
    which lands close enough for preview rendering."""
    try:
        if mat is not None and getattr(mat, "use_nodes", False) and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    inp = node.inputs.get("Base Color")
                    if inp is not None and not inp.is_linked:
                        c = inp.default_value
                        return (float(c[0]) ** 2.0, float(c[1]) ** 2.0,
                                float(c[2]) ** 2.0)
                    break
        c = mat.diffuse_color
        return (float(c[0]) ** 2.0, float(c[1]) ** 2.0, float(c[2]) ** 2.0)
    except Exception:
        return (0.8, 0.8, 0.82)


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
            self._target_size = (width, height)
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
        # orbit/zoom in the viewport reach the renderer live. Re-publish when
        # anything visible changed (camera, world, lights, settings) - not
        # just when the camera moved.
        try:
            self._target_size = (width, height)
            sig = self._scene_sig(depsgraph)
            if self._geom_dirty or sig != self._last_cam_sig:
                view_override = None
                rv3d = getattr(context, "region_data", None)
                if (rv3d is not None and
                        rv3d.view_perspective != 'CAMERA'):
                    view_override = self._viewport_camera(context, width, height)
                self._sync_scene(depsgraph, view_override)
                self._last_cam_sig = sig
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
            status = "Rstr2 v%s | live core | %s" % (
                ".".join(str(x) for x in bl_info["version"]), self._dbg)
        else:
            pixels = _make_test_pattern(width, height)
            status = "Rstr2 v%s | waiting | %s | %s" % (
                ".".join(str(x) for x in bl_info["version"]),
                self._core_msg, self._dbg)

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
        # Route the LINEAR framebuffer through Blender's own color management
        # (scene view transform: Standard / Filmic / AgX + exposure) instead of
        # baking a display transform into the renderer.
        try:
            self.bind_display_space_shader(depsgraph.scene)
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
            self.unbind_display_space_shader()
        except Exception:
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
        gpu.state.blend_set("NONE")

        # Keep redrawing while the engine is active so the live feed updates.
        self.tag_redraw()


    # ------------------------------------------------------------------
    # Phase 3 scene bridge helpers
    # ------------------------------------------------------------------
    def _viewport_camera(self, context, width, height):
        """Camera dict from the current 3D viewport orbit (free navigation).

        Blender orbits around view_location at view_distance along -Z of
        view_rotation; the eye sits one distance behind the pivot. Vertical
        FOV derives from the viewport lens (36mm horizontal sensor fit)."""
        import math
        from mathutils import Vector

        rv3d = getattr(context, "region_data", None)
        if rv3d is None:
            return None
        q = rv3d.view_rotation
        forward = q @ Vector((0.0, 0.0, -1.0))
        right = q @ Vector((1.0, 0.0, 0.0))
        up = q @ Vector((0.0, 1.0, 0.0))
        pivot = Vector(rv3d.view_location)
        origin = pivot - forward * float(rv3d.view_distance)
        lens = 50.0
        sd = getattr(context, "space_data", None)
        if sd is not None and getattr(sd, "lens", 0):
            lens = float(sd.lens)
        aspect = max(float(width), 1.0) / max(float(height), 1.0)
        tan_half_v = (18.0 / lens) / aspect   # 36mm sensor -> 18mm half-width
        return {
            "origin": tuple(origin),
            "right": (right.x, right.y, right.z),
            "up": (up.x, up.y, up.z),
            "forward": (forward.x, forward.y, forward.z),
            "tan_half_fov_y": tan_half_v,
        }

    def _scene_sig(self, depsgraph):
        """Signature of everything that should trigger a re-publish: camera,
        world light, render settings and light objects. Compared each redraw
        so edits like World color/strength reach the core live."""
        sig = self._camera_sig(depsgraph)
        extras = []
        world = _world_light(depsgraph.scene)
        if world is not None:
            extras += [round(v, 4) for v in world]
        sprops = getattr(depsgraph.scene, "rstr2", None)
        extras += [
            bool(getattr(sprops, "use_taa", True)),
            round(float(getattr(sprops, "exposure", 1.0)), 4),
            int(getattr(sprops, "taa_history", 20)),
            bool(getattr(depsgraph.scene.render, "film_transparent", False)),
        ]
        lights = []
        for obj in depsgraph.objects:
            if obj.type != "LIGHT":
                continue
            mw = obj.matrix_world
            lights += [
                round(float(mw[0][3]), 3), round(float(mw[1][3]), 3),
                round(float(mw[2][3]), 3),
                str(obj.data.type),
                round(float(getattr(obj.data, "energy", 0)), 2),
                round(float(getattr(obj.data, "color", (0, 0, 0))[0]), 3),
                round(float(getattr(obj.data, "color", (0, 0, 0))[1]), 3),
                round(float(getattr(obj.data, "color", (0, 0, 0))[2]), 3),
                round(float(getattr(obj.data, "spot_size", 0)), 4),
                round(float(getattr(obj.data, "size", 0)), 3),
            ]
        return sig + tuple(extras + lights)

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

    def _sync_scene(self, depsgraph, view_override=None):
        """Extract meshes + camera from the depsgraph and publish to the core."""
        tw, th = getattr(self, "_target_size", None) or (0, 0)
        aspect = (float(tw) / float(th)) if (tw and th) else (16.0 / 9.0)
        scene = _extract_scene(depsgraph, view_override, aspect)
        if scene is None:
            self._dbg = "no mesh"
            return
        if not self._ensure_scene_writer():
            self._dbg = "scene map n/a"
            return
        try:
            vertices, indices, camera, lights, albedos = scene

            # Scene-level render settings (blRstr-parity panel + film).
            sprops = getattr(depsgraph.scene, "rstr2", None)
            sflags = 1 if getattr(sprops, "use_taa", True) else 0
            if getattr(depsgraph.scene.render, "film_transparent", False):
                sflags |= 2  # FLAG_FILM_TRANSPARENT
            settings = {
                "flags": sflags,
                "exposure": float(getattr(sprops, "exposure", 1.0)),
                "history": float(getattr(sprops, "taa_history", 20)),
                # Render at the actual target size so the image matches the
                # viewport outline / F12 resolution (no aspect stretching).
                "size": list(getattr(self, "_target_size", None) or (0, 0)),
            }
            world = _world_light(depsgraph.scene)
            if world is not None:
                settings["world"] = list(world)

            ok = self._scene_writer.write(vertices, indices, camera, lights,
                                          albedos, settings)
            self._dbg = ("scene %d v / %d i / %d L%s"
                         % (vertices.shape[0], indices.shape[0],
                            lights.shape[0],
                            "" if albedos.size else " no-alb")) if ok else "scene write fail"
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


# ----------------------------------------------------------------------
# Scene-level render settings (blRstr-parity) + Render-properties panel
# ----------------------------------------------------------------------
class Rstr2SceneSettings(bpy.types.PropertyGroup):
    use_taa: bpy.props.BoolProperty(
        name="Use TAA",
        description="Temporal anti-aliasing + accumulation "
                    "(jittered rays + exponential history)",
        default=True,
    )
    exposure: bpy.props.FloatProperty(
        name="Exposure",
        description="Pre-tonemap exposure multiplier",
        min=0.05, max=10.0, default=1.0,
    )
    taa_history: bpy.props.IntProperty(
        name="TAA History Length",
        description="Frames of temporal history. Higher = smoother, "
                    "slower to react",
        min=1, max=128, default=20,
    )


class RSTR2_PT_settings(bpy.types.Panel):
    bl_label = "Rstr2"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        if context.scene.render.engine != "RSTR2":
            layout.label(text="Select the Rstr2 render engine", icon="INFO")
            return
        props = getattr(context.scene, "rstr2", None)
        if props is None:
            return
        col = layout.column(align=True)
        col.prop(props, "use_taa")
        if props.use_taa:
            col.prop(props, "taa_history")
        col = layout.column()
        col.prop(props, "exposure")


def register():
    bpy.utils.register_class(Rstr2Engine)
    bpy.utils.register_class(Rstr2SceneSettings)
    bpy.utils.register_class(RSTR2_PT_settings)
    bpy.types.Scene.rstr2 = bpy.props.PointerProperty(type=Rstr2SceneSettings)


def unregister():
    del bpy.types.Scene.rstr2
    bpy.utils.unregister_class(RSTR2_PT_settings)
    bpy.utils.unregister_class(Rstr2SceneSettings)
    bpy.utils.unregister_class(Rstr2Engine)


if __name__ == "__main__":
    register()
