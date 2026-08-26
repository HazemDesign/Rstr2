// Rstr2 - OptiX kernel: ReSTIR DI (RTXDI-style direct lighting) + TAA.
//
// Pass 1 (rg_primary): jittered primary ray -> G-buffer (world pos, normal,
// per-vertex albedo), then build a per-pixel light reservoir by RIS-sampling
// candidate lights from a typed pool (point / sun / spot / area) and combine
// it with the previous frame's reservoir (temporal reuse). The exact
// stochastic light-sample position is carried inside the reservoir so the
// shading pass evaluates the same sample.
//
// Pass 2 (rg_shade): read the G-buffer + reservoir, cast ONE shadow ray for
// the selected light sample, accumulate linear HDR color temporally
// (exponential moving average = TAA), then tonemap to the display buffer.
// Phase 5: after the direct term, spawn cosine-weighted secondary rays for
// global illumination. Each secondary hit is written to a dedicated bounce
// buffer (so the primary G-buffer survives) and shaded with a 1-spp light
// estimate; a miss reaches the environment (world radiance).
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
static __device__ __forceinline__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
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
// Componentwise product (CUDA has no float3*float3 - it miscompiles; see
// AGENTS.md "hard-won constraints").
static __device__ __forceinline__ float3 mul3(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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

// ---- RNG -------------------------------------------------------------------
static __device__ __forceinline__ uint32_t rng_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
static __device__ __forceinline__ float rng_f(uint32_t& s) {
    uint32_t x = rng_next(s);
    return (float)(x & 0x00FFFFFFu) / (float)0x01000000u;
}

static __device__ __forceinline__ uint32_t pixel_seed(unsigned int pidx,
                                                      unsigned int frame,
                                                      unsigned int salt) {
    uint32_t s = (pidx + 1u) * 9781u + (frame + 1u) * 26699u + salt * 6271u;
    s ^= s >> 15;
    return s;
}

// ---- typed light sampling --------------------------------------------------
// Light types (must match rstr2::Light doc):
#define LIGHT_POINT 0u
#define LIGHT_SUN   1u
#define LIGHT_SPOT  2u
#define LIGHT_AREA  3u

// Uniform direction on the unit sphere (Marsaglia via rejection-free z<->1).
static __device__ __forceinline__ float3 rand_unit_vec(uint32_t& seed) {
    float z = rng_f(seed) * 2.0f - 1.0f;
    float a = rng_f(seed) * 6.2831853f;
    float r = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    return make_float3(r * cosf(a), r * sinf(a), z);
}

// Spot cone falloff: smoothstep between cos_outer and cos_inner evaluated at
// the cosine between the light's emission axis and the emission direction at
// the shaded point (-Ld, pointing from light toward surface).
static __device__ __forceinline__ float spot_factor(const rstr2::Light& L,
                                                     const float3& Ld) {
    float cosA = dot3(Ld * -1.0f, normalize3(make_f3(L.dx, L.dy, L.dz)));
    float co = L.size_x;   // cos(outer half angle)
    float ci = L.size_y;   // cos(inner half angle)
    if (ci <= co) return (cosA >= co) ? 1.0f : 0.0f;   // blend == 0 hard edge
    float t = (cosA - co) / (ci - co);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Sample one stochastic position on the light. Returns:
//   sp   - world-space sample position (unused for sun)
//   Ld   - unit direction from P toward the sample (or toward the sun)
//   dist - distance to the sample (1e8 for sun)
//   G    - unshadowed geometry/importance term p_hat used for RIS weights
static __device__ void sample_light(const rstr2::Light& L, const float3& P,
                                    const float3& N, uint32_t& seed,
                                    float3& sp, float3& Ld, float& dist,
                                    float& G) {
    uint32_t type = (uint32_t)(L.type + 0.5f);
    if (type == LIGHT_SUN) {
        // Directional. size_x carries Blender's sun `angle` (radians): jitter
        // the direction inside that cone for soft shadows.
        Ld = normalize3(make_f3(-L.dx, -L.dy, -L.dz));
        if (L.size_x > 0.0f) {
            float3 t1 = normalize3(cross3(Ld, fabsf(Ld.z) < 0.9f
                                                ? make_f3(0.0f, 0.0f, 1.0f)
                                                : make_f3(1.0f, 0.0f, 0.0f)));
            float3 t2 = cross3(Ld, t1);
            float ang = L.size_x * 0.5f * rng_f(seed);
            float phi = rng_f(seed) * 6.2831853f;
            Ld = normalize3(Ld + (t1 * (cosf(phi)) + t2 * sinf(phi))
                                      * tanf(ang));
        }
        dist = 1e8f;
        sp = P + Ld;
        float ndl = fmaxf(dot3(N, Ld), 0.0f);
        G = ndl;                      // no distance falloff
        return;
    }

    float3 lp = make_f3(L.px, L.py, L.pz);
    if (type == LIGHT_AREA) {
        // Uniform random point on the emitting rectangle.
        float u = rng_f(seed) - 0.5f;
        float v = rng_f(seed) - 0.5f;
        float3 dir = normalize3(make_f3(L.dx, L.dy, L.dz));
        float3 ax = normalize3(make_f3(L.ax, L.ay, L.az));
        float3 ay = cross3(dir, ax);   // unit (dir _|_ ax)
        sp = lp + ax * (u * L.size_x) + ay * (v * L.size_y);
    } else if (type == LIGHT_SPOT && L.az > 0.0f) {
        // Blender spot: az carries shadow_soft_size -> spherical source.
        sp = lp + rand_unit_vec(seed) * L.az;
    } else if (type == LIGHT_POINT && L.ax > 0.0f) {
        // Blender point light: ax carries shadow_soft_size.
        sp = lp + rand_unit_vec(seed) * L.ax;
    } else {
        sp = lp;                        // point & spot: fixed position
    }

    float3 toL = sp - P;
    float d2 = dot3(toL, toL);
    if (d2 < 1e-8f) { toL = N; d2 = 1.0f; }
    float d = sqrtf(d2);
    Ld = toL * (1.0f / d);
    dist = d;
    float ndl = fmaxf(dot3(N, Ld), 0.0f);
    G = ndl / fmaxf(d2, 1e-6f);
    if (type == LIGHT_SPOT) G *= spot_factor(L, Ld);
}

// ---- camera -----------------------------------------------------------------
static __device__ __forceinline__ float3 camera_ray(float px, float py) {
    float u = (px + 0.5f + params.jitter_x) / (float)params.width;
    float v = (py + 0.5f + params.jitter_y) / (float)params.height;
    float aspect = (float)params.width / (float)params.height;
    float uvx = (2.0f * u - 1.0f) * params.cam_tan_half_fov_y * aspect
              + params.cam_shift_x * 2.0f * params.cam_tan_half_fov_y * aspect;
    float uvy = (1.0f - 2.0f * v) * params.cam_tan_half_fov_y
              + params.cam_shift_y * 2.0f * params.cam_tan_half_fov_y; // row 0 = top
    float3 dir = v3(params.cam_forward) + v3(params.cam_right) * uvx + v3(params.cam_up) * uvy;
    return normalize3(dir);
}

// ---- cosine-weighted hemisphere sampling (Lambertian bounce) ----------------
static __device__ float3 cosine_sample_hemisphere(const float3& N, uint32_t& seed) {
    float3 up = (fabsf(N.z) < 0.999f) ? make_f3(0.0f, 0.0f, 1.0f)
                                      : make_f3(1.0f, 0.0f, 0.0f);
    float3 t = normalize3(cross3(up, N));
    float3 b = cross3(N, t);
    float r1 = rng_f(seed);
    float r2 = rng_f(seed);
    float phi = 6.2831853f * r1;
    float r = sqrtf(r2);
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - r2));
    return normalize3(t * x + b * y + N * z);
}

// Single-sample direct lighting at an arbitrary point (used for GI bounces).
// Picks one light uniformly (prob 1/NL); the per-sample weight for a uniform
// pick is NL * G, matching the ReSTIR DI estimator form used for the primary
// hit so the two agree. Returns f_r * Le * vis * (NL*G) for the single sample.
static __device__ float3 estimate_direct_1spp(const float3& P, const float3& N,
                                              const float3& albedo,
                                              uint32_t& seed) {
    uint32_t NL = params.light_count;
    if (NL == 0u) return make_f3(0.0f, 0.0f, 0.0f);
    uint32_t li = rng_next(seed) % NL;
    rstr2::Light L = params.lights[li];
    float3 sp, Ld;
    float dist, G;
    sample_light(L, P, N, seed, sp, Ld, dist, G);
    if (G <= 0.0f) return make_f3(0.0f, 0.0f, 0.0f);

    // Ld/dist come straight from sample_light (already jittered for sun/point/
    // spot soft shadows and correct for area emitters).
    float tmax = fmaxf(dist - 2.0f * 1e-3f, 1e-3f);
    uint32_t q0 = 1u, q1 = 0u, q2 = 0u, q3 = 0u;
    optixTrace(params.handle, P + N * 1e-3f, Ld, 1e-3f, tmax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0u, 1u, 0u, q0, q1, q2, q3);
    float vis = (float)q1;
    if (vis <= 0.0f) return make_f3(0.0f, 0.0f, 0.0f);

    float3 Le = make_f3(L.cr, L.cg, L.cb) * L.intensity;
    float3 f_r = albedo * (1.0f / 3.14159265f);
    float W = (float)NL * G;
    return make_float3(W * f_r.x * Le.x, W * f_r.y * Le.y, W * f_r.z * Le.z);
}

// ============================ MISS ==========================================
extern "C" __global__ void __miss__ms() {
    uint32_t rt = optixGetPayload_0();
    if (rt == 1u) {
        // Shadow ray reached the sky with no occluder -> visible.
        optixSetPayload_1(1u);
    } else if (rt == 2u) {
        // Secondary (GI) ray reached the environment -> no surface hit.
        optixSetPayload_1(0u);
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

    // Geometric normal + material albedo from the triangle's soup vertices.
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

    // Per-vertex albedo (triangle soup => all 3 verts share the material).
    float3 alb = make_f3(0.8f, 0.8f, 0.82f);
    if (params.albedos) {
        rstr2::Vec3F a = params.albedos[i0];
        alb = make_f3(a.x, a.y, a.z);
    }

    if (rt == 2u) {
        // Secondary (GI) hit: write to the per-pixel bounce buffer so the
        // primary G-buffer (used for the final present) is preserved. The
        // shading pass reads this after the trace and continues the path.
        float4* bb = (float4*)params.bounce_buf;
        bb[3u * pidx + 0u] = make_float4(P.x, P.y, P.z, 1.0f);
        bb[3u * pidx + 1u] = make_float4(ng.x, ng.y, ng.z, 0.0f);
        bb[3u * pidx + 2u] = make_float4(alb.x, alb.y, alb.z, 0.0f);
        optixSetPayload_1(1u);
        return;
    }

    float4* g = (float4*)params.gbuf;
    g[3u * pidx + 0u] = make_float4(P.x, P.y, P.z, 1.0f);
    g[3u * pidx + 1u] = make_float4(ng.x, ng.y, ng.z, 0.0f);
    g[3u * pidx + 2u] = make_float4(alb.x, alb.y, alb.z, 0.0f);
}

// ---- world / environment ----------------------------------------------------
static __device__ __forceinline__ float3 world_radiance() {
    if (params.world_strength > 0.0f) {
        return make_float3(params.world_r * params.world_strength,
                           params.world_g * params.world_strength,
                           params.world_b * params.world_strength);
    }
    return make_f3(0.03f, 0.04f, 0.06f);   // legacy default backdrop
}

// ---- temporal display accumulation (TAA) ------------------------------------
static __device__ __forceinline__ void accumulate_and_present(
    unsigned int pidx, const float3& hdr, float out_alpha) {
    // Firefly clamp before blending (blRstr-style "TAA clamping").
    float3 c = hdr;
    if (params.taa_clamp > 0.0f) {
        c.x = fminf(c.x, params.taa_clamp);
        c.y = fminf(c.y, params.taa_clamp);
        c.z = fminf(c.z, params.taa_clamp);
    }
    // Same memory layout as Vec4F; reinterpret to use CUDA vector ops.
    float4* accBuf = (float4*)params.accum;
    float4 prev = accBuf[pidx];
    float a = params.accum_alpha;
    float3 acc = make_float3(prev.x + (c.x - prev.x) * a,
                             prev.y + (c.y - prev.y) * a,
                             prev.z + (c.z - prev.z) * a);
    accBuf[pidx] = make_float4(acc.x, acc.y, acc.z, out_alpha);

    // LINEAR scene-referred output: exposure only. Display transform
    // (Standard/Filmic/AgX) is applied by Blender's color management.
    float3 disp = acc * params.exposure;
    float4* img = (float4*)params.image;
    img[pidx] = make_float4(disp.x, disp.y, disp.z, out_alpha);
}

// ====================== RAYGEN: primary + reservoir ========================
extern "C" __global__ void __raygen__rg_primary() {
    uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;
    unsigned int pidx = idx.x + idx.y * params.width;

    float4* g = (float4*)params.gbuf;
    g[3u * pidx].w = 0.0f; // mark no-hit until closest hit writes it

    float3 origin = v3(params.cam_origin);
    float3 dir = camera_ray((float)idx.x, (float)idx.y);

    uint32_t p0 = 0u, p1 = 0u, p2 = 0u, p3 = 0u;
    optixTrace(params.handle, origin, dir, 0.0f, 1e16f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0u, 1u, 0u, p0, p1, p2, p3);

    rstr2::Reservoir r;
    r.wsum = 0.0f; r.lightIdx = 0u; r.M = 0u; r.pad = 0u;
    r.sx = 0.0f; r.sy = 0.0f; r.sz = 0.0f;

    float4 ph = g[3u * pidx];
    if (ph.w > 0.5f) { // valid surface hit
        float3 P = make_f3(ph.x, ph.y, ph.z);
        float3 N = make_f3(g[3u * pidx + 1u].x, g[3u * pidx + 1u].y, g[3u * pidx + 1u].z);

        uint32_t NL = params.light_count;
        if (NL > 0u) {
            uint32_t s = pixel_seed(pidx, params.frame_index, 1u);
            const int CANDIDATES = 8;
            const uint32_t MAX_M = (1u << 20);
            for (int i = 0; i < CANDIDATES; ++i) {
                uint32_t li = rng_next(s) % NL;
                rstr2::Light L = params.lights[li];

                float3 sp, Ld;
                float dist, G;
                sample_light(L, P, N, s, sp, Ld, dist, G);
                if (G <= 0.0f) continue;

                float wi = (float)NL * G;    // w_i = p_hat / p, p = 1/NL
                float wnew = r.wsum + wi;
                r.wsum = wnew;
                r.M += 1u;
                if (rng_f(s) < wi / fmaxf(wnew, 1e-12f)) {
                    r.lightIdx = li;
                    r.sx = sp.x; r.sy = sp.y; r.sz = sp.z;
                }
            }

            // Temporal reuse: combine with previous frame's reservoir.
            rstr2::Reservoir pr = params.prev_reservoirs[pidx];
            if (pr.M > 0u && pr.lightIdx < NL) {
                float total = r.wsum + pr.wsum;
                if (rng_f(s) < pr.wsum / fmaxf(total, 1e-12f)) {
                    r.lightIdx = pr.lightIdx;
                    r.sx = pr.sx; r.sy = pr.sy; r.sz = pr.sz;
                }
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
    float4 ph = g[3u * pidx];

    if (ph.w < 0.5f) {
        // Background / environment (accumulated so silhouettes converge).
        float alpha = params.film_transparent ? 0.0f : 1.0f;
        accumulate_and_present(pidx, world_radiance(), alpha);
        return;
    }

    float3 P = make_f3(ph.x, ph.y, ph.z);
    float3 N = make_f3(g[3u * pidx + 1u].x, g[3u * pidx + 1u].y, g[3u * pidx + 1u].z);
    float3 albedo = make_f3(g[3u * pidx + 2u].x, g[3u * pidx + 2u].y, g[3u * pidx + 2u].z);

    uint32_t NL = params.light_count;

    // Fallback shading when no analytic lights are provided (default scene).
    if (NL == 0u) {
        float3 ambient = make_f3(0.0f, 0.0f, 0.0f);
        if (params.world_strength > 0.0f) {
            float ws = params.world_strength;
            ambient = make_float3(albedo.x * ws * params.world_r,
                                  albedo.y * ws * params.world_g,
                                  albedo.z * ws * params.world_b);
        }
        float3 key = normalize3(make_f3(0.3f, 0.7f, -0.6f));
        float ndl = max(dot3(N, key), 0.0f);
        float3 col = albedo * (0.25f + 0.75f * ndl) + ambient;
        accumulate_and_present(pidx, col, 1.0f);
        return;
    }

    // ---- Primary direct lighting via the ReSTIR DI reservoir. ----
    float3 L = make_f3(0.0f, 0.0f, 0.0f);
    rstr2::Reservoir r = params.reservoirs[pidx];
    if (r.M > 0u && r.lightIdx < NL && r.wsum > 0.0f) {
        float W = r.wsum / max((float)r.M, 1.0f);
        rstr2::Light Ll = params.lights[r.lightIdx];
        uint32_t type = (uint32_t)(Ll.type + 0.5f);
        float3 Ld;
        float tmax;
        if (type == LIGHT_SUN) {
            Ld = normalize3(make_f3(-Ll.dx, -Ll.dy, -Ll.dz));
            tmax = 1e8f;
        } else {
            float3 sp = make_f3(r.sx, r.sy, r.sz);
            float3 toS = sp - P;
            float ds = length3(toS);
            Ld = (ds > 1e-6f) ? toS * (1.0f / ds) : N;
            tmax = fmaxf(ds - 2.0f * 1e-3f, 1e-3f);
        }
        uint32_t p0 = 1u, p1 = 0u, p2 = 0u, p3 = 0u;
        optixTrace(params.handle, P + N * 1e-3f, Ld, 1e-3f, tmax, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   0u, 1u, 0u, p0, p1, p2, p3);
        float vis = (float)p1;
        float3 Le = make_f3(Ll.cr, Ll.cg, Ll.cb) * Ll.intensity;
        float3 f_r = albedo * (1.0f / 3.14159265f);
        L = make_float3(W * f_r.x * Le.x * vis,
                        W * f_r.y * Le.y * vis,
                        W * f_r.z * Le.z * vis);
    }

    // ---- Indirect global illumination bounces (Phase 5). ----
    // Lambertian throughput: each diffuse bounce multiplies throughput by the
    // surface albedo (the ndl / pdf cosine terms cancel). The environment is
    // reached when a secondary ray misses (multiplied by current throughput).
    float3 thr = make_f3(1.0f, 1.0f, 1.0f);
    float3 curP = P, curN = N, curAlb = albedo;
    uint32_t seed = pixel_seed(pidx, params.frame_index, 7u);
    for (uint32_t b = 0u; b < params.max_bounces; ++b) {
        float3 dir = cosine_sample_hemisphere(curN, seed);
        uint32_t p0 = 2u, p1 = 0u, p2 = 0u, p3 = 0u;
        optixTrace(params.handle, curP + curN * 1e-3f, dir, 1e-3f, 1e16f, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   0u, 1u, 0u, p0, p1, p2, p3);
        if (p1 == 0u) {
            L += mul3(thr, world_radiance());   // env contribution along this path
            break;
        }
        float4* bb = (float4*)params.bounce_buf;
        float3 P2 = make_f3(bb[3u * pidx + 0u].x, bb[3u * pidx + 0u].y, bb[3u * pidx + 0u].z);
        float3 N2 = make_f3(bb[3u * pidx + 1u].x, bb[3u * pidx + 1u].y, bb[3u * pidx + 1u].z);
        float3 alb2 = make_f3(bb[3u * pidx + 2u].x, bb[3u * pidx + 2u].y, bb[3u * pidx + 2u].z);

        thr = mul3(thr, curAlb);
        float3 direct2 = estimate_direct_1spp(P2, N2, alb2, seed);
        L += mul3(thr, direct2);

        curP = P2; curN = N2; curAlb = alb2;
    }

    accumulate_and_present(pidx, L, 1.0f);
}
