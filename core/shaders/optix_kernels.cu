// Rstr2 - OptiX kernel. Single-bounce triangle soup path tracer that mirrors
// the previous DXR shader (raytracing.hlsl): same camera math (Y-flipped NDC,
// aspect-correct FOV), geometric normal shading with a fixed key light.
//
// The host renderer passes the scene through a single global `Params params`,
// bound by the pipeline as the launch-parameter variable ("params"). The
// per-ray color is carried in three 32-bit payload registers (p0..p2) to stay
// correct on 64-bit platforms (a pointer would not fit in one register).
//
// NOTE: We deliberately avoid CUDA's built-in float3 operators and the
// vector overloads of dot/normalize/cross (they are not reliably available
// when nvcc is forced past its supported host compiler). All vector math is
// implemented explicitly below.

#include <optix.h>
#include <cuda_runtime.h>

#include "optix_params.h"

using namespace rstr2;

// Launch parameters provided by the host. Bound by optixLaunch. Declared in
// the global namespace (name "params") so the pipeline's
// pipelineLaunchParamsVariableName = "params" resolves to this symbol.
// OptiX requires the launch-parameter variable to live in CONSTANT memory, so
// it must be __constant__ (not __device__); OptiX binds the launch-param
// buffer to this symbol at optixLaunch time.
__constant__ rstr2::Params params;

// Vec3F <-> float3 reinterpretation (identical 12-byte layout).
static __forceinline__ __device__ float3 to_float3(const Vec3F& v) {
    return make_float3(v.x, v.y, v.z);
}

// --- Explicit float3 math (no reliance on CUDA vector operator overloads) ---
static __forceinline__ __device__ float3 vneg(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}
static __forceinline__ __device__ float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __device__ float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __device__ float3 vabs3(float3 a) {
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}
static __forceinline__ __device__ float3 smul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __device__ float3 sadd(float3 a, float s) {
    return make_float3(a.x + s, a.y + s, a.z + s);
}
static __forceinline__ __device__ float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __device__ float3 vcross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}
static __forceinline__ __device__ float3 vnorm(float3 a) {
    return smul(a, rsqrtf(vdot(a, a)));
}

extern "C" __global__ void __raygen__rg() {
    uint3 idx = optixGetLaunchIndex();

    unsigned int pixel_idx = idx.y * params.width + idx.x;

    // uv in [0,1]; flip Y so the first row of the buffer is the TOP of the image.
    // (Scalar math only: CUDA's float2/float3 operator overloads are not
    // reliably available with the unsupported host compiler, so we avoid them.)
    float fx = ((float)idx.x + 0.5f) / (float)params.width;
    float fy = ((float)idx.y + 0.5f) / (float)params.height;
    float ndcx = fx * 2.0f - 1.0f;
    float ndcy = -(fy * 2.0f - 1.0f);

    float aspect = (float)params.width / (float)params.height;

    float3 origin = to_float3(params.cam_origin);
    float3 fwd    = to_float3(params.cam_forward);
    float3 right  = to_float3(params.cam_right);
    float3 up     = to_float3(params.cam_up);

    float3 dir = vnorm(vadd(vadd(fwd,
                          smul(right, ndcx * aspect * params.cam_tan_half_fov_y)),
                          smul(up, ndcy * params.cam_tan_half_fov_y)));

    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);
    optixTrace(
        params.handle,
        origin, dir,
        0.001f, 1e16f, 0.0f,
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2);

    float3 color = make_float3(__uint_as_float(p0),
                               __uint_as_float(p1),
                               __uint_as_float(p2));
    params.image[pixel_idx] = Vec4F{ color.x, color.y, color.z, 1.0f };
}

extern "C" __global__ void __miss__ms() {
    // Dark background.
    optixSetPayload_0(__float_as_uint(0.03f));
    optixSetPayload_1(__float_as_uint(0.05f));
    optixSetPayload_2(__float_as_uint(0.09f));
}

extern "C" __global__ void __closesthit__ch() {
    unsigned int prim = optixGetPrimitiveIndex();
    Vec3F* v = params.vertices;
    unsigned int* id = params.indices;
    unsigned int i0 = id[prim * 3 + 0];
    unsigned int i1 = id[prim * 3 + 1];
    unsigned int i2 = id[prim * 3 + 2];

    float3 p0 = to_float3(v[i0]);
    float3 p1 = to_float3(v[i1]);
    float3 p2 = to_float3(v[i2]);

    // Geometric normal (vertices already world space).
    float3 N = vnorm(vcross(vsub(p1, p0), vsub(p2, p0)));
    float3 rd = optixGetWorldRayDirection();
    if (vdot(N, rd) > 0.0f) N = vneg(N); // face the camera

    // Phase 3 shading: normal-tinted base + fixed key light.
    // (Many lights / ReSTIR DI arrive in a later phase.)
    float3 base = sadd(smul(vabs3(N), 0.55f), 0.20f);
    float3 lightDir = vnorm(make_float3(0.4f, 0.8f, -0.3f));
    float ndl = fmaxf(vdot(N, lightDir), 0.0f);
    float3 color = smul(base, 0.25f + 0.75f * ndl);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}
