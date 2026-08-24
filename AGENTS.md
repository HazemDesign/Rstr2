# AGENTS.md — Rstr2 native DXR core

Rstr2Core is the native (C++/D3D12/DXR) ray-tracing engine that the Blender
addon drives over shared memory. It is Phase 3: a single-bounce DXR 1.0
renderer that ray-traces a world-space triangle soup and publishes RGBA32F
frames. This file records build/debug knowledge that is easy to lose.

## Layout
- `core/src/main.cpp` — entry point. Default mode inits DXR on a worker
  thread with an explicit **3 GB stack** (some D3D12 driver paths consume a
  huge amount of stack per call; the OS `/STACK` PE flag is not always
  honoured, so the thread is created with a large stack to avoid
  `STATUS_STACK_OVERFLOW`). `--ci` prints version and exits 0 (no GPU).
- `core/src/renderer.cpp` — device/adapter selection, resource + accel
  creation, the DXR state object (`create_pipeline`), SBT (`create_sbt`),
  and the per-frame `DispatchRays` (`render_frame`).
- `core/src/shared_mem.cpp` — shared-memory bridge: `SharedMem` (frame
  writer) and `SceneMem` (scene reader).
- `core/shaders/raytracing.hlsl` — shader model 6.5 `lib_6_5` library with
  `raygenMain` / `missMain` / `hitMain` and the `HitGroup` hit group.

## Build
All native builds happen in CI (`.github/workflows/build.yml`,
`windows-latest`). The workflow:
1. Downloads a pinned `dxc` (DirectXShaderCompiler) and compiles
   `raytracing.hlsl` → `raytracing.cso`.
2. Fetches the Agility SDK NuGet `Microsoft.Direct3D.D3D12` 1.619.0 and
   builds with `AGILITY_SDK_DIR` set.
3. Uploads an artifact `Rstr2Core-win64` containing `Rstr2Core.exe`,
   `raytracing.cso`, and a `D3D12/` subfolder with `D3D12Core.dll` +
   `d3d12SDKLayers.dll`.

Local builds also work (Windows SDK + dxc) but do **not** ship the Agility
`D3D12/` folder, so the OS `d3d12.dll` is used instead of the side-loaded
runtime.

## Debug layer (IMPORTANT gotcha)
To get precise D3D12/DXR validation errors (e.g. from `CreateStateObject`),
run with `RSTR2_DEBUG=1`. The app writes diagnostics to `Rstr2Core.log`
(stderr is redirected there by the launcher).

The debug layer is provided via the **Agility SDK side-loading** mechanism,
not the OS "Graphics Tools" feature. Reasons:
- On this class of machine the OS `d3d12.dll` (e.g. 8972) and the Graphics
  Tools `d3d12sdklayers.dll` (e.g. 1591) are version-mismatched, so enabling
  the OS debug layer makes `D3D12CreateDevice` fail.
- `d3d12.dll` is **not** side-loadable. Only `D3D12Core.dll` is, and only
  when the EXE exports `D3D12SDKVersion` and `D3D12SDKPath`. That is what
  `core/src/agility_sdk.cpp` does (guarded by `RSTR2_AGILITY_SDK`, defined
  when `AGILITY_SDK_DIR` is set). CMake copies the SDK DLLs into a `D3D12/`
  **subfolder** next to the exe and the export points there. Co-locating the
  DLLs directly next to the exe (instead of a subfolder) breaks the loader
  and yields `0x887E0003` (`D3D12_ERROR_INVALID_REDIST`).

GPU-Based Validation is intentionally disabled at runtime (it overflows the
stack during resource/descriptor creation); the plain debug layer is enough
to catch state-object and binding errors.

## Bugs that were fixed (keep these constraints in mind)
1. **`CreateStateObject` → `E_INVALIDARG` (0x80070057).** The shader
   declared `gOutput` as `RWBuffer<float4>` (a *typed* UAV) at `u0`, but the
   global root signature binds `u0` as a **root-descriptor UAV**. D3D12 only
   permits Raw/Structured buffers on root-descriptor UAVs. Fix: use
   `RWStructuredBuffer<float4>` in `raytracing.hlsl`. Rule of thumb: a typed
   UAV (`RWTexture2D`, `RWBuffer`) needs a *descriptor-table* UAV entry, not
   a root descriptor.
2. **Runtime crash `0xC0000005` (ACCESS_VIOLATION) after the first
   `DispatchRays`.** The readback buffer was created with
   `D3D12_HEAP_TYPE_DEFAULT`, which cannot be `Map()`-ed for CPU reads; the
   `memcpy` from the returned (null/garbage) pointer AV'd. Fix: the readback
   buffer uses `D3D12_HEAP_TYPE_READBACK`.

## Notes
- The SBT references the hit group by its export name `HitGroup` (not the
  closest-hit shader `hitMain` it imports) in `create_sbt`.
- The shader config association in `create_pipeline` lists
  `{raygenMain, missMain, hitMain}`; the DXR spec permits associating with
  individual component shaders, so this is valid.
- `render_frame` binds the TLAS SRV via a descriptor-table root parameter
  (slot 0, in the shader-visible heap) and everything else as root
  descriptors (vertices/index/camera/output).
