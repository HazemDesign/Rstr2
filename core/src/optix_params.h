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

// Typed analytic light. Flat 16-float (64-byte) layout so host/device/SHM
// agree without struct-alignment surprises. Radiance = color * intensity.
//   type 0 point: shaded from (px,py,pz), isotropic.
//   type 1 sun:   directional; shines along (dx,dy,dz); no distance falloff.
//   type 2 spot:  point + cone around (dx,dy,dz); size_x = cos(outer half
//                 angle), size_y = cos(inner half angle) (soft edge between).
//   type 3 area:  rectangular emitter centered at (px,py,pz), facing along
//                 (dx,dy,dz), extents size_x/size_y along unit tangent (ax..az)
//                 and cross(direction, axis). Sampled stochastically.
struct Light {
    float px, py, pz;     // world position (point/spot/area)
    float type;           // see enum above
    float dx, dy, dz;     // emission direction (sun/spot/area), normalized
    float intensity;      // multiplier on color
    float cr, cg, cb;    // linear RGB radiance (pre-intensity)
    float size_x;         // spot: cos_outer | area: x extent
    float size_y;         // spot: cos_inner | area: y extent
    float ax, ay, az;     // area: first tangent axis (unit length)
};

// ReSTIR DI reservoir for a single pixel. Stores the summed RIS weight, the
// selected light index, the number of candidates folded in (M), and the exact
// stochastic light-sample position the weight was computed with (required so
// the shading pass evaluates the SAME sample for area lights). Temporal
// reuse combines two of these additively, carrying the winning sample.
struct Reservoir {
    float    wsum;       // sum of w_i = p_hat/p over candidates
    uint32_t lightIdx;   // selected light
    uint32_t M;          // total candidates considered (across frames)
    uint32_t pad;
    float    sx, sy, sz; // world-space sample position of the selected light
};

// Host-side definition of the launch parameters. The device mirrors this with
// a global `Params params;` whose variable name is bound by the pipeline
// (OPTIX pipelineLaunchParamsVariableName = "params").
struct Params {
    Vec4F*        image;            // device ptr, RGBA32F tonemapped output,
                                    // row-major, row0 = TOP
    unsigned int  width;
    unsigned int  height;
    Vec3F*        vertices;         // device ptr, world-space xyz triples
    unsigned int* indices;          // device ptr, uint32 triangle indices
    Vec3F*        albedos;          // device ptr, per-vertex linear RGB or null
    Vec3F         cam_origin;
    Vec3F         cam_right;
    Vec3F         cam_up;
    Vec3F         cam_forward;
    float         cam_tan_half_fov_y;
    uint64_t      handle;           // OptixTraversableHandle (single GAS)

    // --- RTXDI (ReSTIR DI direct lighting) state ---
    Light*        lights;           // device ptr, typed-light pool
    unsigned int  light_count;      // number of lights (0 => fallback shading)
    Vec4F*        gbuf;             // 3*N float4: [pos.xyz,hit],
                                    //             [normal.xyz,_],
                                    //             [albedo.rgb,_]
    Reservoir*    reservoirs;       // current-frame reservoirs (write)
    Reservoir*    prev_reservoirs;  // previous-frame reservoirs (read)
    unsigned int  frame_index;      // increments each frame (for RNG + reuse)

    // --- TAA / display state ---
    Vec4F*        accum;            // HDR accumulation buffer (N float4)
    float         jitter_x;         // subpixel jitter in pixels, [-0.5, 0.5)
    float         jitter_y;
    float         accum_alpha;      // EMA blend for the new sample (1 = reset)
    float         exposure;         // pre-tonemap multiplier
    float         taa_clamp;        // firefly clamp before EMA (0 = off)

    // --- World / film ---------------------------------------------------
    float         world_r, world_g, world_b;  // uniform env radiance
    float         world_strength;             // 0 disables world light
    unsigned int  film_transparent;           // alpha=0 for background pixels
};

} // namespace rstr2
