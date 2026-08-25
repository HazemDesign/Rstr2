// Rstr2 Phase 3 - OptiX renderer.
//
// Ray-traces a world-space triangle mesh supplied by the Blender addon (via
// the scene shared-memory bridge) with a camera also supplied by the addon.
// Falls back to a single hardcoded triangle when no scene is set.
//
// API-compatible with the previous DXR renderer so the rest of the core
// (main.cpp, shared_mem) is unchanged.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shared_mem.h"

namespace rstr2 {

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Initialize the OptiX/CUDA device, default scene, pipeline and SBT.
    // Returns false on any failure and fills `error`.
    bool init(int width, int height, std::string& error);

    // Render a single frame. `out_pixels` must hold width*height*4 floats.
    bool render_frame(float* out_pixels, std::string& error);

    // Replace the scene (geometry + camera) and rebuild acceleration
    // structures. Safe to call between frames from the render loop.
    bool set_scene(const SceneData& scene, std::string& error);

    // Reallocate per-pixel buffers for a new frame size (viewport/F12).
    // Pipeline, SBT and acceleration structures are unaffected.
    bool resize(int width, int height, std::string& error);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;

    int width_ = 0;
    int height_ = 0;

    SceneData scene_;
    bool have_scene_ = false;
};

} // namespace rstr2
