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
- [ ] **Phase 2** - Interop: native exe initializes D3D12, ray-traces a hardcoded
      triangle, ships pixels over shared memory; addon displays them.
- [ ] **Phase 3** - Scene sync: meshes/camera/lights from depsgraph -> DXR accel structures.
- [ ] **Phase 4** - Many lights + ReSTIR DI.
- [ ] **Phase 5** - Temporal accumulation/denoise, indirect bounces.

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

## References

- Malt (bnpr/Malt) - external-process viewport streaming pattern
- RadeonProRenderBlenderAddon - minimal texture-upload/display pattern
- Blender manual: custom render engines / gpu module
