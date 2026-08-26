# AGENTS.md — Rstr2 (OptiX core + Blender addon)

Rstr2 is a from-scratch real-time ray traced render engine for Blender
4.x/5.x, inspired by [blRstr](https://stkrake.net/blRstr/) (ReSTIR/RTXDI
many-light rendering). Same architecture as blRstr: a Python addon launches a
separate native renderer process and talks to it over Windows shared memory.

The native core is **OptiX** (NVIDIA), not DXR anymore: an early DXR 1.0/D3D12
implementation was replaced by OptiX 9 + CUDA driver API (see git history).
It renders ReSTIR DI direct lighting (typed light pool + temporal reservoir
reuse) with TAA accumulation and publishes linear RGBA32F frames.

## Layout

```
addon/
  __init__.py    RenderEngine subclass "RSTR2": view_draw/view_update/render,
                 scene extraction (_extract_scene/_world_light/_material_rgb),
                 liveness signature (_scene_sig), viewport-camera override,
                 Rstr2 settings panel (TAA/exposure/history)
  core_proc.py   Locates + launches bin/Rstr2Core.exe headless, log redirect
  shm.py         Frame bridge reader  (Local\Rstr2Frame_v1,  core -> addon)
  scene_shm.py   Scene bridge writer  (Local\Rstr2Scene_v1, addon -> core)
core/
  src/main.cpp        Entry point, arg parsing (--ci/--width/--height/--shm),
                      worker thread with EXPLICIT 3 GB stack, 30 fps poll loop,
                      frame shm sized for max(2048x1280, CLI size)
  src/renderer.cpp    CUDA/OptiX init, GAS build, pipeline+SBT, set_scene,
                      resize, two-pass optixLaunch per frame
  src/shared_mem.*    SharedMem (frame writer) + SceneMem (scene reader)
  src/optix_params.h  Params/Light/Reservoir structs shared host<->device
  shaders/optix_kernels.cu  nvcc -> PTX; rg_primary + rg_shade raygen passes
tools/update-addon.ps1  Installs addon/*.py + bin/{exe,ptx} into Blender's
                        addons dir (run with Blender CLOSED)
tests/smoke_register.py Headless registration smoke test (Blender --background)
bin/               Rstr2Core.exe + optix_kernels.ptx consumed by the addon
                   (CI artifact; not tracked in git)
```

## Build

All native builds happen in CI (`.github/workflows/build.yml`,
`windows-2022` runner - NOT windows-latest: VS2026 is too new for CUDA 12.5):

1. `Jimver/cuda-toolkit` installs CUDA 12.5.0 (nvcc).
2. OptiX 9.0 headers come from a sparse checkout of `NVIDIA/optix-dev`
   (headers only; `optix.64.dll` is loaded from the NVIDIA driver at runtime).
3. CMake (`-DOPTIX_INCLUDE_DIR=...`) builds C++20 **CXX-only** - the CMake
   CUDA language is deliberately NOT enabled (it would demand a VS2026 CUDA
   toolset that CUDA 12.5 doesn't ship). The kernel becomes PTX via a custom
   `nvcc -ptx --gpu-architecture=compute_70 -allow-unsupported-compiler`
   command and is copied next to the exe.
4. Links `CUDA::cuda_driver` with `/DELAYLOAD:nvcuda.dll` so the CI `--ci`
   smoke run starts on machines without an NVIDIA driver.
5. Artifact `Rstr2Core-win64` = `Rstr2Core.exe` + `optix_kernels.ptx`.
   Unzip both into `bin/`.

CI triggers only on changes under `core/**`, `addon/**`, or the workflow file;
empty commits elsewhere do NOT trigger it.

Addon install: run `powershell -ExecutionPolicy Bypass -File
tools\update-addon.ps1` (copies py files + core binaries into
`%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\rstr2`; the addon
finds the exe in `<installed addon>/bin` first, then `<repo>/bin`, then
`RSTR2_CORE_BIN`). Or symlink the addon folder for auto-update dev flow.

## Runtime data flow

1. `view_draw` computes a scene signature (`_scene_sig` = camera basis + FOV +
   world color/strength + rstr2 settings + film_transparent + every light's
   pos/type/energy/color/spot_size/size). Any change republishes the whole
   scene over `Local\Rstr2Scene_v1` (epoch bumped LAST, `writing=1` guard).
2. When `rv3d.view_perspective != 'CAMERA'`, `_viewport_camera` derives eye =
   pivot - forward*view_distance from the orbit quaternion (36 mm sensor ->
   tan_half_v = (18/lens)/aspect) so free navigation renders what you see.
   F12 always uses `depsgraph.scene.camera` (with shift_x/y + sensor fit).
3. Core polls the scene mapping every ~33 ms, applies requested render size
   (clamped 16..shm max, reallocs output/gbuf/reservoirs/accum via
   `Renderer::resize` and zeroes temporal state), then `set_scene` rebuilds
   geometry/lights and flags `scene_dirty`.
4. Each `render_frame`: pass 1 `rg_primary` (jittered primary rays -> G-buffer
   pos/normal/albedo + ReSTIR DI reservoir w/ temporal reuse), pass 2
   `rg_shade` (one shadow ray, EMA accumulate into HDR buffer, exposure scale,
   LINEAR output). Ping-pong reservoir buffers by frame_index&1.
5. Frame published to `Local\Rstr2Frame_v1` (RGBA32F, row 0 = TOP,
   frame_index incremented LAST); addon reads with read/copy/re-read retry,
   uploads GPUTexture, draws fullscreen quad through
   `bind_display_space_shader` so Blender's own view transform (Standard /
   Filmic / AgX) does display mapping - the core ships linear values.
6. Fallback everywhere: if the core is missing/fails, the animated numpy test
   pattern keeps drawing; no addon path may raise into Blender.

## Protocol v2 (keep Python `scene_shm.py` and `shared_mem.h` in sync!)

Scene header = 256 B @ offset 0, `_pack_(4)`:
`magic 0x32525353, version 2, epoch, ready, writing, vertex_count,
index_count, light_count, flags(bit0 TAA, bit1 filmTransparent), exposure,
taa_history, cam_origin/right/up/forward[3], cam_tan_half_fov_y, reserved[]`.
Blocks after the header, in order:
vertices (n*3 f32 world space) | indices (m u32) | lights (L * 16 f32) |
albedos (n*3 f32).

Reserved-area addendum (packed by the writer, parsed by the reader):
- bytes 0..15 : f32 world_color[3] + world_strength (0 => no world light)
- bytes 16..23: u32 render_width, render_height (0 = keep current)
- bytes 24..31: f32 camera shift_x, shift_y

Typed-light row (16 f32 / 64 B):
`[px,py,pz, type, dx,dy,dz, intensity, cr,cg,cb, size_x,size_y, ax,ay,az]`
types: 0 point, 1 sun, 2 spot, 3 area. Spot: size_x = cos(outer half angle),
size_y = cos(inner half angle). Sun: size_x carries Blender sun `angle`
(radians, soft-shadow cone jitter). Area: extents along axis (ax..az) and
cross(dir, axis). Point/Spot carry shadow_soft_size in ax / az respectively.

**CRITICAL CONTRACT:** the albedo block is ALWAYS present - the core reads it
unconditionally. If a writer omits it, subsequent blocks shift and surfaces
read stale zeros (black render with intact background - this exact bug shipped
once). The Python writer substitutes 0.8-gray rows when the caller has none.
Geometry uses a NON-indexed triangle soup (one vertex per triangle corner) so
per-triangle material albedo stays exact across material boundaries.

## Hard-won constraints (do not regress)

- **CUDA device code: never multiply float3*float3** - write it componentwise
  (`.x*.x + ...`). A `float3*float3` silently miscompiles/won't compile and
  cost a debugging round trip once (commit 7040c4f).
- `optixTerminateRay` is illegal in closest-hit used for shadow rays; use
  payload flags instead (removed in 2940b73).
- Visibility/shadow rays need proper ray flags/masks - an earlier
  `OPTIX_RAY_FLAG_NONE` made all rays miss (a9db424).
- Worker thread runs with an explicit 3 GB stack (`_beginthreadex`): some
  driver paths consume huge stack per call and the PE `/STACK` flag is not
  reliably honoured; avoids `STATUS_STACK_OVERFLOW`.
- Blender 5.x: use Principled BSDF **Base Color**, not `diffuse_color`
  (non-node materials ignore edits to diffuse_color at runtime);
  `bmesh.ops` plane creation and `Object.look_at` helpers don't exist there.
- sRGB->linear approximation everywhere in the addon is gamma-2.0 (`c*c`) -
  good enough for preview, kept consistent between materials and world color.
- The viewport camera basis taken straight from matrix_world columns is
  correct (an earlier manual derivation produced a mirrored image).
- TAA reset: `accum_alpha = 1.0` on the first frame after any scene change
  (`scene_dirty`) or when TAA is off; otherwise `1/history` EMA with a 10.0
  firefly clamp. Jitter only when accumulating.
- Phase 5 indirect GI (in `optix_kernels.cu`): `rg_shade` spawns
  cosine-weighted secondary rays with ray-type `2` (`rt==2`). `__closesthit__`
  routes those hits to `params.bounce_buf` (3 float4/pixel, same layout as
  `gbuf`) so the primary G-buffer survives; a miss on `rt==2` is the
  environment. `renderer.cpp`'s `maxTraceDepth` must stay `>= max_bounces + 2`
  (bounce + its shadow). `kMaxBounces` (1 = one indirect bounce) sets
  `params.max_bounces`; set to 0 for direct-only. The `/DELAYLOAD` + CI build
  path is the only place the `.cu`->PTX->exe core can be compiled (no local
  CUDA toolkit), so kernel changes are verified by pushing to GitHub Actions.

## Debugging

- Everything the core prints goes to stderr, which `core_proc.py` redirects to
  `bin/Rstr2Core.log` (timestamped lines; launch failures visible there).
- Viewport status line (top-left of the render) shows version + live/waiting +
  last sync/draw error - check it first ("scene err", "no mesh",
  "frame map not open (core exited, code N)", etc.).
- After each scene change the core dumps launch params + a center probe pixel
  (linear + accumulated) to the log for the first frames.
- CI smoke: `build\Release\Rstr2Core.exe --ci` must print the version and exit
  0 without touching the GPU.

## Validation matrix (all green as of e950f33)

point-light distance falloff | sun uniform lighting | spot cone + soft
penumbra edge | area-light soft gradient | world ambient scales with strength
| film transparent => alpha 0 background | distinct per-material albedo |
TAA bit-stable when nothing moves | resize keeps rendering at new viewport/F12
size | free-nav orbit matches viewport outline.
