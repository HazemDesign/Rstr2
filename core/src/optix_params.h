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

// Point light. Flat 8-float layout (32 bytes) so host/device/SHM agree without
// any struct-alignment surprises. World-space position, radiance = color *
// intensity, and a pad slot for future use (e.g. radius/area).
struct PointLight {
    float px, py, pz;     // world position
    float intensity;      // multiplier on color
    float cr, cg, cb;    // linear RGB radiance (pre-intensity)
    float pad;
};

// ReSTIR DI reservoir for a single pixel. Stores the summed RIS weight, the
// selected light index, and the number of candidates folded in (M). Temporal
// reuse combines two of these additively.
struct Reservoir {
    float    wsum;       // sum of w_i = p_hat/p over candidates
    uint32_t lightIdx;   // selected light
    uint32_t M;          // total candidates considered (across frames)
    uint32_t pad;
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

    // --- RTXDI (ReSTIR DI direct lighting) state ---
    PointLight*   lights;           // device ptr, point-light pool
    unsigned int  light_count;      // number of lights (0 => fallback shading)
    Vec4F*        gbuf;             // device ptr, 2*N float4: [pos.xyz, hitFlag],
                                    //                        [normal.xyz, _]
    Reservoir*    reservoirs;       // device ptr, current-frame reservoirs (write)
    Reservoir*    prev_reservoirs;  // device ptr, previous-frame reservoirs (read)
    unsigned int  frame_index;      // increments each frame (for RNG + reuse)
};

} // namespace rstr2
