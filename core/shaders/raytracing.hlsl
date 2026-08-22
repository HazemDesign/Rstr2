// Rstr2 Phase 3 - DXR ray tracing shader library (shader model 6.5, lib_6_5).
//
// Ray-traces a world-space triangle soup supplied by the native core: the
// BLAS/TLAS are built from the scene bridge, and the camera basis arrives via
// a constant buffer. Output is written to a RWBuffer<float4> (RGBA32F) at
// linear index (y * width + x); the first row of the buffer is the TOP.

RWBuffer<float4> gOutput : register(u0);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<float3> Vertices : register(t1);
StructuredBuffer<uint> Indices : register(t2);

cbuffer Camera : register(b0) {
    float3 camOrigin;
    float3 camRight;
    float3 camUp;
    float3 camForward;
    float  camTanHalfFovY;
};

struct RayPayload {
    float4 color;
};

struct HitAttribs {
    float2 bary;
};

[shader("raygeneration")]
void raygenMain() {
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    // uv in [0,1]; flip Y so the first row of the buffer is the TOP of the image.
    float2 uv = (float2(idx) + 0.5f) / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    float aspect = float(dim.x) / float(dim.y);
    float3 dir = normalize(
        camForward +
        ndc.x * aspect * camTanHalfFovY * camRight +
        ndc.y * camTanHalfFovY * camUp);

    RayDesc ray;
    ray.Origin = camOrigin;
    ray.Direction = dir;
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    RayPayload payload;
    payload.color = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // TraceRay(AS, flags, mask, rayContribution, multiplier, missIndex, ray, payload)
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    gOutput[idx.y * dim.x + idx.x] = payload.color;
}

[shader("miss")]
void missMain(inout RayPayload payload) {
    // Dark background.
    payload.color = float4(0.03f, 0.05f, 0.09f, 1.0f);
}

[shader("closesthit")]
void hitMain(inout RayPayload payload, in HitAttribs attribs) {
    uint prim = PrimitiveIndex();
    uint i0 = Indices[prim * 3 + 0];
    uint i1 = Indices[prim * 3 + 1];
    uint i2 = Indices[prim * 3 + 2];

    float3 p0 = Vertices[i0];
    float3 p1 = Vertices[i1];
    float3 p2 = Vertices[i2];

    // Geometric normal (vertices are already in world space).
    float3 N = normalize(cross(p1 - p0, p2 - p0));
    float3 rd = WorldRayDirection();
    if (dot(N, rd) > 0.0f) N = -N; // face the camera

    // Simple shading for Phase 3: normal-tinted base with a fixed key light.
    // (Many lights / ReSTIR DI arrive in Phase 4.)
    float3 base = abs(N) * 0.55f + 0.20f;
    float3 lightDir = normalize(float3(0.4f, 0.8f, -0.3f));
    float ndl = max(dot(N, lightDir), 0.0f);
    float3 color = base * (0.25f + 0.75f * ndl);

    payload.color = float4(color, 1.0f);
}
