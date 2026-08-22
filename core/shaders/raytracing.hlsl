// Rstr2 Phase 2 - DXR ray tracing shader library (shader model 6.3, lib_6_3).
//
// Renders ONE hardcoded triangle (3 vertices in the z=0 plane) with simple
// Lambert shading. Output is written to a RWBuffer<float4> (RGBA32F) which the
// native core copies into Win32 shared memory for the Blender addon to read.
//
// NOTE: the spec text said "RWTexture2D gOutput". A buffer resource cannot be
// bound to RWTexture2D, so we declare it as RWBuffer<float4> and index it
// linearly (y * width + x). This is the only deviation required to make the
// "output DEFAULT+RW buffer" requirement actually work.

RWBuffer<float4> gOutput : register(u0);
RaytracingAccelerationStructure SceneBVH : register(t0);

// Per-hit vertex buffer, supplied via the hit group's local root signature
// (root SRV, register t0, space1).
StructuredBuffer<float3> Vertices : register(t0, space1);

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
    float tanHalf = tan(0.45f); // ~half FOV

    // Camera at (0, 0.6, -2.2). The triangle lives in the z=0 plane, so the
    // camera must look toward +z to actually hit it. (The brief said "looking
    // -z"; taken literally that would miss the triangle entirely and render a
    // black frame, so we use +z to produce a visible result.)
    float3 origin = float3(0.0f, 0.6f, -2.2f);
    float3 dir = normalize(float3(ndc.x * aspect * tanHalf, ndc.y * tanHalf, 1.0f));

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    RayPayload payload;
    payload.color = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // TraceRay(AS, flags, mask, rayContribution, multiplier, missIndex, ray, payload)
    TraceRay(SceneBVH, RAY_FLAG_NONE, 1, 0, 1, 0, ray, payload);

    gOutput[idx.y * dim.x + idx.x] = payload.color;
}

[shader("miss")]
void missMain(inout RayPayload payload) {
    // Dark background.
    payload.color = float4(0.03f, 0.05f, 0.09f, 1.0f);
}

[shader("closesthit")]
void hitMain(inout RayPayload payload, in HitAttribs attribs) {
    float3 bary = float3(
        1.0f - attribs.bary.x - attribs.bary.y,
        attribs.bary.x,
        attribs.bary.y);

    // Per-vertex base colors: red / green / blue.
    float3 c0 = float3(0.90f, 0.20f, 0.20f);
    float3 c1 = float3(0.20f, 0.90f, 0.20f);
    float3 c2 = float3(0.20f, 0.40f, 0.90f);
    float3 base = bary.x * c0 + bary.y * c1 + bary.z * c2;

    // Geometric normal from the vertex buffer (planar triangle).
    float3 e1 = Vertices[1] - Vertices[0];
    float3 e2 = Vertices[2] - Vertices[0];
    float3 N = normalize(cross(e1, e2));
    float3 rd = WorldRayDirection();
    if (dot(N, rd) > 0.0f) N = -N; // face the camera

    float3 lightDir = normalize(float3(0.4f, 0.8f, -0.3f));
    float ndl = max(dot(N, lightDir), 0.0f);

    float3 color = base * (0.2f + 0.8f * ndl); // ambient + Lambert
    payload.color = float4(color, 1.0f);
}
