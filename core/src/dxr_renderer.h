// Rstr2 - DXR 1.1 renderer backend.
//
// Same public interface and scene contract as optix renderer.cpp, but traced
// with DirectX Raytracing 1.1 (cross-vendor: runs on any DXR-capable GPU, not
// just NVIDIA). Reuses shared_mem.cpp (SceneMem reader + SharedMem frame
// writer) verbatim, so the addon and GPU integration tests are unchanged.
//
// Behavior intentionally mirrors optix_kernels.cu (direct lighting over the
// typed light pool + one cosine-weighted GI bounce + environment, EMA temporal
// accumulation) so output is numerically comparable.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shared_mem.h"

namespace rstr2 {

class DxrRenderer {
public:
    DxrRenderer();
    ~DxrRenderer();

    DxrRenderer(const DxrRenderer&) = delete;
    DxrRenderer& operator=(const DxrRenderer&) = delete;

    bool init(int width, int height, std::string& error);
    bool render_frame(float* out_pixels, std::string& error);
    bool set_scene(const SceneData& scene, std::string& error);
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
