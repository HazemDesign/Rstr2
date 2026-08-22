# Rstr2

A from-scratch real-time ray traced render engine for Blender 4.x/5.x,
inspired by blRstr (ReSTIR/RTXDI many-light rendering). Built for an
RTX 5050 on Windows 10/11 with **no local Visual Studio install** -
the native core is compiled by GitHub Actions and consumed as a small artifact.

## Architecture

```
Blender (Python addon)                 Native core (C++/HLSL, built by CI)
├─ addon/__init__.py                   ├─ core/src/*
│  RenderEngine subclass               │  D3D12 device + DXR 1.1
│  scene sync                          │  BLAS/TLAS accel structs
│  shared memory client      ◄────────►│  shared memory server
│  viewport draw (GPUTexture)          │  ReSTIR DI resampling
└─ F12 path (numpy .rect)              └─ temporal accumulation/denoise
```

## Phases

- [x] **Phase 1** - Addon skeleton: registers "Rstr2" engine, animated test pattern
       in viewport (`view_draw`) and F12 (`render`). Proves the post-bgl framebuffer path.
- [x] **Phase 2** - Interop: native exe initializes D3D12/DXR, ray-traces a hardcoded
       triangle, ships RGBA32F pixels over shared memory (`Local\Rstr2Frame_v1`);
       addon displays them. Built by GitHub Actions (no local VS).
- [x] **Phase 3** - Scene sync: addon extracts world-space triangle soup + camera basis
       from the depsgraph and publishes it over `Local\Rstr2Scene_v1`; the core polls
       it, rebuilds BLAS/TLAS each update, and ray-traces the real geometry with the
       supplied camera (falls back to the hardcoded triangle when no scene is sent).
- [ ] **Phase 4** - Many lights + ReSTIR DI.
- [ ] **Phase 5** - Temporal accumulation/denoise, indirect bounces.

> Note: Phases 2-3 compile and pass the CI build/smoke. End-to-end GPU rendering
> (actual DXR dispatch on an RTX 5050) is verified by running the addon locally with
> the downloaded `bin/Rstr2Core.exe` artifact.

## Install the addon locally (Blender 5.2)

Option A - symlink (auto-updates while developing):

```powershell
$dst = "$env:APPDATA\Blender Foundation\Blender\5.2\scripts\addons\rstr2"
New-Item -ItemType SymbolicLink -Path $dst -Target "D:\blender\dev\Rstr2\addon" -Force
```

Option B - zip it: compress the `addon/` folder to `rstr2.zip`,
then Edit > Preferences > Add-ons > Install.

Then: Render Properties > Render Engine > **Rstr2**, switch the 3D viewport to
Material Preview or Rendered - you should see an animated gradient test pattern.

## Build the native core (GitHub Actions, no local VS)

Push to GitHub. The `build-core` workflow compiles `core/` on a Windows runner
and uploads `Rstr2Core-win64` as an artifact (a few MB).
Download it from the repo's Actions tab > latest run > Artifacts.

Unzip it to `bin/` in this repo:

```
D:\blender\dev\Rstr2\bin\Rstr2Core.exe
```

The addon will look for the core there by default (configurable in addon
preferences once Phase 2 wires up the shared-memory bridge).

## References

- Malt (bnpr/Malt) - external-process viewport streaming pattern
- RadeonProRenderBlenderAddon - minimal texture-upload/display pattern
- Blender manual: custom render engines / gpu module
