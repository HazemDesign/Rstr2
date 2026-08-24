// Rstr2 - OptiX launch parameters (shared between host renderer and the
// OptiX kernel). Plain C-compatible layout so the same struct is readable
// from both the .cu device code and the host C++. No OptiX/CUDA types are
// used here so the host can include this without pulling in CUDA headers.

#pragma once

#include <cstdint>

namespace rstr2 {

// Minimal 3/4-component float vectors. Layout matches CUDA float3/float4 so
// the kernel can reinterpret to the CUDA vector types without copying.
struct Vec3F {
    float x, y, z;
};
struct Vec4F {
    float x, y, z, w;
};

// Host-side definition of the launch parameters. The device mirrors this with
// a global `Params params;` whose variable name is bound by the pipeline
// (OPTIX pipelineLaunchParamsVariableName = "params").
struct Params {
    Vec4F*        image;            // device ptr, RGBA32F, row-major, row0 = TOP
    unsigned int  width;
    unsigned int  height;
    Vec3F*        vertices;         // device ptr, world-space xyz triples
    unsigned int* indices;          // device ptr, uint32 triangle indices
    Vec3F         cam_origin;
    Vec3F         cam_right;
    Vec3F         cam_up;
    Vec3F         cam_forward;
    float         cam_tan_half_fov_y;
    uint64_t      handle;           // OptixTraversableHandle (single GAS)
};

} // namespace rstr2
