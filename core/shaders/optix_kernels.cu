// Rstr2 - OptiX kernel: ReSTIR DI (RTXDI-style direct lighting).
//
// Pass 1 (rg_primary): trace the primary ray, write a G-buffer (world pos +
// normal + hit flag), then build a per-pixel light reservoir by RIS-sampling
// candidate lights and combining it with the previous frame's reservoir
// (temporal reuse).
//
// Pass 2 (rg_shade): read the G-buffer + reservoir, cast ONE shadow ray for
// the selected light, and write the lit color.
//
// The miss/closest-hit programs are shared and branch on the ray-type payload
// (0 = primary, 1 = shadow).

#include <optix.h>

#include "optix_params.h"

// ---- optix device builtins are provided by optix.h -------------------------

// Globals bound by the pipeline (variable name "params").
__constant__ rstr2::Params params;

// ---- minimal float3 helpers (self-contained, no cuda headers) --------------
static __device__ __forceinline__ float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}
static __device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __device__ __forceinline__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __device__ __forceinline__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
static __device__ __forceinline__ float length3(const float3& a) {
    return sqrtf(dot3(a, a));
}
static __device__ __forceinline__ float3 normalize3(const float3& a) {
    float l = length3(a);
    return (l > 1e-8f) ? a * (1.0f / l) : make_float3(0.0f, 0.0f, 1.0f);
}
static __device__ __forceinline__ float3 v3(const rstr2::Vec3F& v) {
    return make_float3(v.x, v.y, v.z);
}
// Narkowicz ACES filmic approximation, then sRGB gamma encode.
static __device__ __forceinline__ float3 tonemap(const float3& x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float3 t = make_float3(
        (x.x * (a * x.x + b)) / (x.x * (c * x.x + d) + e),
        (x.y * (a * x.y + b)) / (x.y * (c * x.y + d) + e),
        (x.z * (a * x.z + b)) / (x.z * (c * x.z + d) + e));
    return make_float3(powf(fmaxf(t.x, 0.0f), 1.0f / 2.2f),
                       powf(fmaxf(t.y, 0.0f), 1.0f / 2.2f),
                       powf(fmaxf(t.z, 0.0f), 1.0f / 2.2f));
}

// ---- RNG -------------------------------------------------------------------
static __device__ __forceinline__ uint32_t rng_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
static __device__ __forceinline__ float rng_f(uint32_t& s) {
    uint32_t x = rng_next(s);
    return (float)(x & 0x00FFFFFFu) / (float)0x01000000u;
}

static __device__ __forceinline__ uint32_t pixel_seed(unsigned int pidx, unsigned int frame) {
    uint32_t s = (pidx + 1u) * 9781u + (frame + 1u) * 26699u;
    s ^= s >> 15;
    return s;
}

// ---- ray generation (shared camera basis) -----------------------------------
static __device__ __forceinline__ float3 camera_ray(unsigned int px, unsigned int py) {
    float u = ((float)px + 0.5f) / (float)params.width;
    float v = ((float)py + 0.5f) / (float)params.height;
    float aspect = (float)params.width / (float)params.height;
    float uvx = (2.0f * u - 1.0f) * params.cam_tan_half_fov_y * aspect;
    float uvy = (1.0f - 2.0f * v) * params.cam_tan_half_fov_y; // row 0 = top
    float3 dir = v3(params.cam_forward) + v3(params.cam_right) * uvx + v3(params.cam_up) * uvy;
    return normalize3(dir);
}

// ============================ MISS ==========================================
extern "C" __global__ void __miss__ms() {
    uint32_t rt = optixGetPayload_0();
    if (rt == 1u) {
        // Shadow ray reached the light with no occluder -> visible.
        optixSetPayload_1(1u);
    }
    // Primary miss: nothing to do (G-buffer hit flag stays 0).
}

// ====================== CLOSEST HIT =========================================
extern "C" __global__ void __closesthit__ch() {
    uint32_t rt = optixGetPayload_0();
    if (rt == 1u) {
        // Shadow ray hit geometry -> occluded. Returning from the closest-hit
        // program ends the ray (no any-hit program is used for shadows).
        optixSetPayload_1(0u);
        return;
    }

    unsigned int pidx = optixGetLaunchIndex().x +
                         optixGetLaunchIndex().y * params.width;

    // World-space hit position from the ray.
    float3 ro = optixGetWorldRayOrigin();
    float3 rd = optixGetWorldRayDirection();
    float  t  = optixGetRayTmax();
    float3 P  = ro + rd * t;

    // Geometric normal from the triangle's world vertices.
    unsigned int prim = optixGetPrimitiveIndex();
    const rstr2::Vec3F* v = params.vertices;
    const unsigned int* idx = params.indices;
    unsigned int i0 = idx[3u * prim + 0u];
    unsigned int i1 = idx[3u * prim + 1u];
    unsigned int i2 = idx[3u * prim + 2u];
    float3 va = make_f3(v[i0].x, v[i0].y, v[i0].z);
    float3 vb = make_f3(v[i1].x, v[i1].y, v[i1].z);
    float3 vc = make_f3(v[i2].x, v[i2].y, v[i2].z);
    float3 ng = normalize3(cross3(vb - va, vc - va));

    float4* g = (float4*)params.gbuf;
    g[2u * pidx]     = make_float4(P.x, P.y, P.z, 1.0f);
    g[2u * pidx + 1u] = make_float4(ng.x, ng.y, ng.z, 0.0f);
}

// ====================== RAYGEN: primary + reservoir ========================
extern "C" __global__ void __raygen__rg_primary() {
    uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;
    unsigned int pidx = idx.x + idx.y * params.width;

    float4* g = (float4*)params.gbuf;
    g[2u * pidx].w = 0.0f; // mark no-hit until closest hit writes it

    float3 origin = v3(params.cam_origin);
    float3 dir = camera_ray(idx.x, idx.y);

    uint32_t p0 = 0u, p1 = 0u, p2 = 0u, p3 = 0u;
    optixTrace(params.handle, origin, dir, 0.0f, 1e16f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0u, 1u, 0u, p0, p1, p2, p3);

    rstr2::Reservoir r;
    r.wsum = 0.0f; r.lightIdx = 0u; r.M = 0u; r.pad = 0u;

    float4 ph = g[2u * pidx];
    if (ph.w > 0.5f) { // valid surface hit
        float3 P = make_f3(ph.x, ph.y, ph.z);
        float3 N = make_f3(g[2u * pidx + 1u].x, g[2u * pidx + 1u].y, g[2u * pidx + 1u].z);

        uint32_t NL = params.light_count;
        if (NL > 0u) {
            uint32_t s = pixel_seed(pidx, params.frame_index);
            const int CANDIDATES = 8;
            const uint32_t MAX_M = (1u << 20);
            for (int i = 0; i < CANDIDATES; ++i) {
                uint32_t li = rng_next(s) % NL;
                rstr2::PointLight L = params.lights[li];
                float3 toL = make_f3(L.px, L.py, L.pz) - P;
                float d2 = dot3(toL, toL);
                if (d2 < 1e-6f) continue;
                float d = sqrtf(d2);
                float3 Ld = toL * (1.0f / d);
                float ndl = max(dot3(N, Ld), 0.0f);
                float G = ndl / d2;             // p_hat (geometry term)
                float wi = (float)NL * G;        // w_i = p_hat / p, p = 1/NL
                r.wsum += wi;
                r.M += 1u;
                if (rng_f(s) < wi / max(r.wsum, 1e-12f)) r.lightIdx = li;
            }

            // Temporal reuse: combine with previous frame's reservoir.
            rstr2::Reservoir pr = params.prev_reservoirs[pidx];
            if (pr.M > 0u) {
                float total = r.wsum + pr.wsum;
                if (rng_f(s) < pr.wsum / max(total, 1e-12f)) r.lightIdx = pr.lightIdx;
                r.wsum = total;
                uint32_t m = pr.M + r.M;
                r.M = (m > MAX_M) ? MAX_M : m;
            }
        }
    }

    params.reservoirs[pidx] = r;
}

// ====================== RAYGEN: shading =====================================
extern "C" __global__ void __raygen__rg_shade() {
    uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;
    unsigned int pidx = idx.x + idx.y * params.width;

    float4* g = (float4*)params.gbuf;
    float4 ph = g[2u * pidx];

    float3 background = make_f3(0.03f, 0.04f, 0.06f);
    float4* img = (float4*)params.image;

    if (ph.w < 0.5f) {
        img[pidx] = make_float4(background.x, background.y, background.z, 1.0f);
        return;
    }

    float3 P = make_f3(ph.x, ph.y, ph.z);
    float3 N = make_f3(g[2u * pidx + 1u].x, g[2u * pidx + 1u].y, g[2u * pidx + 1u].z);

    uint32_t NL = params.light_count;

    // Fallback shading when no lights are provided (default scene).
    if (NL == 0u) {
        float3 base = make_f3(0.85f, 0.85f, 0.88f);
        float3 key = normalize3(make_f3(0.3f, 0.7f, -0.6f));
        float ndl = max(dot3(N, key), 0.0f);
        float3 col = base * (0.25f + 0.75f * ndl);
        float3 mapped = tonemap(col);
        img[pidx] = make_float4(mapped.x, mapped.y, mapped.z, 1.0f);
        return;
    }

    rstr2::Reservoir r = params.reservoirs[pidx];
    float W = r.wsum / max((float)r.M, 1.0f);

    rstr2::PointLight L = params.lights[r.lightIdx];
    float3 toL = make_f3(L.px, L.py, L.pz) - P;
    float d = length3(toL);
    float3 Ld = (d > 1e-6f) ? toL * (1.0f / d) : make_f3(0.0f, 1.0f, 0.0f);

    // Shadow ray toward the selected light.
    uint32_t p0 = 1u, p1 = 0u, p2 = 0u, p3 = 0u;
    const float EPS = 1e-3f;
    optixTrace(params.handle, P + N * EPS, Ld, EPS, max(d - 2.0f * EPS, EPS), 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0u, 1u, 0u, p0, p1, p2, p3);
    uint32_t vis = p1;

    float3 Le = make_f3(L.cr, L.cg, L.cb) * L.intensity;
    float3 albedo = make_f3(0.8f, 0.8f, 0.82f);
    float3 f_r = albedo * (1.0f / 3.14159265f);
    float vis_f = (float)vis;
    float3 color = make_float3(W * f_r.x * Le.x * vis_f,
                               W * f_r.y * Le.y * vis_f,
                               W * f_r.z * Le.z * vis_f);

    float3 mapped = tonemap(color);
    img[pidx] = make_float4(mapped.x, mapped.y, mapped.z, 1.0f);
}
