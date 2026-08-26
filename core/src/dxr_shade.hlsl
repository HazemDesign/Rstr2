// Rstr2 - DXR 1.1 kernel: direct lighting (typed light pool) + 1 bounce GI +
// environment, written to expose the SAME behavior as the OptiX kernel
// (optix_kernels.cu) so the addon and GPU integration tests are unchanged.
//
// Two ray-gen passes (matching OptiX rg_primary / rg_shade):
//   RG_Primary : trace primary ray -> G-buffer (P, N, albedo) in a UAV.
//   RG_Shade   : read G-buffer, compute direct lighting over the light pool
//                (with shadow rays), then one cosine-weighted GI bounce (env on
//                miss), and write a jittered HDR sample to the output UAV.
// Temporal accumulation (TAA) is done on the CPU (EMA) in dxr_renderer.cpp,
// exactly like the OptiX path's accum buffer, before publishing the frame.
//
// Ray types: 0 = primary, 1 = shadow, 2 = secondary (GI).
//
// Geometry/lights arrive as ByteAddressBuffers (row-major float blobs) to match
// the exact SceneMem layout in shared_mem.cpp without HLSL float3 alignment
// surprises. Light stride = 64 bytes (16 floats); vertex/albedo stride = 12.

// ---- resources (global root signature) --------------------------------------
cbuffer SceneCb : register(b0) {
    float4 gCamOrigin;   // xyz
    float4 gCamRight;    // xyz
    float4 gCamUp;       // xyz
    float4 gCamForward;  // xyz
    float4 gP0;          // x tanHalfFovY, y camShiftX, z camShiftY, w exposure
    float4 gP1;          // x width, y height, z lightCount, w worldR
    float4 gP2;          // x worldG, y worldB, z worldStrength, w taaClamp
    float4 gP3;          // x frameIndex, y jitterX, z jitterY, w maxBounces
    float4 gP4;          // x filmTransparent
};

ByteAddressBuffer gVerts   : register(t0);
ByteAddressBuffer gIndices : register(t1);
ByteAddressBuffer gAlbedos : register(t2);
ByteAddressBuffer gLights  : register(t3);
RaytracingAccelerationStructure gAS : register(t4);

RWStructuredBuffer<float4> gGbuf   : register(u0); // 3 float4/pixel: P+hit, N, albedo
RWStructuredBuffer<float4> gBounce : register(u1); // 3 float4/pixel: P+hit, N, albedo
RWStructuredBuffer<float4> gOutput : register(u2); // 1 float4/pixel: hdr.rgb, alpha

// ---- RNG --------------------------------------------------------------------
static uint g_rngState;
uint  rngNext() { g_rngState = g_rngState * 1664525u + 1013904223u; return g_rngState; }
float rngF()    { uint x = rngNext(); return (float)(x & 0x00FFFFFFu) / (float)0x01000000u; }

uint pixelSeed(uint pidx, uint frame, uint salt) {
    uint s = (pidx + 1u) * 9781u + (frame + 1u) * 26699u + salt * 6271u;
    s ^= s >> 15;
    return s;
}

// ---- raw scene accessors ----------------------------------------------------
float3 getVertex(uint i)  { return asfloat(gVerts.Load3(i * 12)); }
uint   getIndex(uint i)   { return gIndices.Load(i * 4); }
float3 getAlbedo(uint i)  { return asfloat(gAlbedos.Load3(i * 12)); }

float lf(uint byteOff) { return asfloat(gLights.Load(byteOff)); }

float3 lightPos(uint li)   { uint b = li * 64;      return float3(lf(b+0),  lf(b+4),  lf(b+8)); }
float  lightType(uint li)  { uint b = li * 64;      return lf(b+12); }
float3 lightDir(uint li)   { uint b = li * 64;      return float3(lf(b+16), lf(b+20), lf(b+24)); }
float  lightInt(uint li)   { uint b = li * 64;      return lf(b+28); }
float3 lightCol(uint li)   { uint b = li * 64;      return float3(lf(b+32), lf(b+36), lf(b+40)); }
float  lightSizeX(uint li) { uint b = li * 64;      return lf(b+44); }
float  lightSizeY(uint li) { uint b = li * 64;      return lf(b+48); }
float3 lightAx(uint li)    { uint b = li * 64;      return float3(lf(b+52), lf(b+56), lf(b+60)); }

float3 lightColor(uint li) { return lightCol(li) * lightInt(li); }

// ---- helpers (mirror optix_kernels.cu) --------------------------------------
float3 v3(float3 v) { return v; }

float3 normalize3(float3 a) {
    float l = length(a);
    return (l > 1e-8f) ? a / l : float3(0, 0, 1);
}
float3 cross3(float3 a, float3 b) {
    return float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
float dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

float3 randUnitVec() {
    float z = rngF() * 2.0f - 1.0f;
    float a = rngF() * 6.2831853f;
    float r = sqrt(max(1.0f - z*z, 0.0f));
    return float3(r*cos(a), r*sin(a), z);
}

float spotFactor(float3 dir, float3 Ld, float co, float ci) {
    float cosA = dot3(Ld * -1.0f, normalize3(dir));
    if (ci <= co) return (cosA >= co) ? 1.0f : 0.0f;
    float t = (cosA - co) / (ci - co);
    t = clamp(t, 0.0f, 1.0f);
    return t*t*(3.0f - 2.0f*t);
}

// Sample a stochastic position on light li. Returns Ld (P->sample), dist, G.
void sampleLight(uint li, float3 P, float3 N, out float3 sp, out float3 Ld,
                 out float dist, out float G) {
    float type = lightType(li);
    float3 lp = lightPos(li);
    float3 ld = lightDir(li);
    float size_x = lightSizeX(li);
    float size_y = lightSizeY(li);
    float3 axv = lightAx(li);
    sp = float3(0,0,0); Ld = float3(0,0,1); dist = 1e8f; G = 0.0f;

    if (type > 0.5f && type < 1.5f) { // sun / directional
        Ld = normalize3(-ld);
        if (size_x > 0.0f) {
            float3 t1 = normalize3(cross3(Ld, abs(Ld.z) < 0.9f ? float3(0,0,1) : float3(1,0,0)));
            float3 t2 = cross3(Ld, t1);
            float ang = size_x * 0.5f * rngF();
            float phi = rngF() * 6.2831853f;
            Ld = normalize3(Ld + (t1*cos(phi) + t2*sin(phi)) * tan(ang));
        }
        dist = 1e8f;
        sp = P + Ld;
        G = max(dot3(N, Ld), 0.0f);
        return;
    }

    if (type > 2.5f && type < 3.5f) { // area
        float u = rngF() - 0.5f;
        float vv = rngF() - 0.5f;
        float3 dir = normalize3(ld);
        float3 axn = normalize3(axv);
        float3 ay = cross3(dir, axn);
        sp = lp + axn * (u * size_x) + ay * (vv * size_y);
    } else if (type > 1.5f && type < 2.5f && axv.z > 0.0f) { // spot, soft (az in ax.z)
        sp = lp + randUnitVec() * axv.z;
    } else if (type < 0.5f && axv.x > 0.0f) { // point, soft (ax in ax.x)
        sp = lp + randUnitVec() * axv.x;
    } else {
        sp = lp;
    }

    float3 toL = sp - P;
    float d2 = dot3(toL, toL);
    if (d2 < 1e-8f) { toL = N; d2 = 1.0f; }
    float d = sqrt(d2);
    Ld = toL / d;
    dist = d;
    float ndl = max(dot3(N, Ld), 0.0f);
    G = ndl / max(d2, 1e-6f);
    if (type > 1.5f && type < 2.5f) G *= spotFactor(ld, Ld, size_x, size_y);
}

float3 cameraRay(float px, float py) {
    float u = (px + 0.5f + gP3.y) / (float)gP1.x;
    float v = (py + 0.5f + gP3.z) / (float)gP1.y;
    float aspect = (float)gP1.x / (float)gP1.y;
    float uvx = (2.0f*u - 1.0f) * gP0.x * aspect + gP0.y * 2.0f * gP0.x * aspect;
    float uvy = (1.0f - 2.0f*v) * gP0.x + gP0.z * 2.0f * gP0.x;
    return normalize3(gCamForward.xyz + gCamRight.xyz * uvx + gCamUp.xyz * uvy);
}

float3 cosineSampleHemisphere(float3 N) {
    float3 up = (abs(N.z) < 0.999f) ? float3(0,0,1) : float3(1,0,0);
    float3 t = normalize3(cross3(up, N));
    float3 b = cross3(N, t);
    float r1 = rngF();
    float r2 = rngF();
    float phi = 6.2831853f * r1;
    float r = sqrt(r2);
    float x = r*cos(phi);
    float y = r*sin(phi);
    float z = sqrt(max(0.0f, 1.0f - r2));
    return normalize3(t*x + b*y + N*z);
}

float3 worldRadiance() {
    if (gP2.z > 0.0f)
        return float3(gP1.w * gP2.z, gP2.x * gP2.z, gP2.y * gP2.z);
    return float3(0.03f, 0.04f, 0.06f);
}

// Direct lighting = sum over all lights of G * f_r * Le (matches the OptiX
// single-sample RIS estimator's expectation; W=G when summing all lights).
float3 directLighting(float3 P, float3 N, float3 alb, inout uint seed) {
    uint NL = (uint)gP1.z;
    if (NL == 0u) return float3(0,0,0);
    float3 acc = float3(0,0,0);
    for (uint li = 0; li < NL; li++) {
        float3 sp, Ld; float dist, G;
        sampleLight(li, P, N, sp, Ld, dist, G);
        if (G <= 0.0f) continue;
        float tmax = max(dist - 2.0f*1e-3f, 1e-3f);
        RayDesc ray;
        ray.Origin = P + N*1e-3f;
        ray.Direction = Ld;
        ray.TMin = 1e-3f;
        ray.TMax = tmax;
        uint vis = 1;
        TraceRay(gAS, RAY_FLAG_NONE, 0xff, 1, 0, 1, ray, vis);
        if (vis == 0) continue;
        float3 Le = lightColor(li);
        float3 fr = alb / 3.14159265f;
        acc += fr * Le * G;
    }
    return acc;
}

void writeSurface(uint pid, float3 P, float3 N, float3 alb, RWStructuredBuffer<float4> buf) {
    buf[pid*3 + 0] = float4(P, 1.0f);
    buf[pid*3 + 1] = float4(N, 0.0f);
    buf[pid*3 + 2] = float4(alb, 0.0f);
}

uint  pid() { return DispatchRaysIndex().y * (uint)gP1.x + DispatchRaysIndex().x; }

// ---- ray-gen ----------------------------------------------------------------
[shader("raygeneration")]
void RG_Primary() {
    uint2 i = DispatchRaysIndex().xy;
    if (i.x >= (uint)gP1.x || i.y >= (uint)gP1.y) return;
    float3 dir = cameraRay((float)i.x, (float)i.y);
    RayDesc ray;
    ray.Origin = gCamOrigin.xyz;
    ray.Direction = dir;
    ray.TMin = 1e-3f;
    ray.TMax = 1e8f;
    uint dummy = 0;
    TraceRay(gAS, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, dummy);
}

[shader("raygeneration")]
void RG_Shade() {
    uint2 i = DispatchRaysIndex().xy;
    uint w = (uint)gP1.x, h = (uint)gP1.y;
    if (i.x >= w || i.y >= h) return;
    uint p = pid();
    float4 g0 = gGbuf[p*3 + 0];
    float3 L = float3(0,0,0);
    float alpha = 1.0f;
    if (g0.w < 0.5f) { // primary miss -> environment
        L = worldRadiance();
        if (gP4.x > 0.5f) alpha = 0.0f; // film transparent -> show through
    } else {
        float3 P = g0.xyz;
        float3 N = gGbuf[p*3 + 1].xyz;
        float3 alb = gGbuf[p*3 + 2].xyz;
        uint seed = pixelSeed(p, (uint)gP3.x, 1u);
        L = directLighting(P, N, alb, seed);
        float3 thr = float3(1,1,1);
        float3 curP = P, curN = N, curAlb = alb;
        uint nb = (uint)gP3.w;
        for (uint b = 0; b < nb; b++) {
            float3 dir = cosineSampleHemisphere(curN);
            RayDesc ray;
            ray.Origin = curP + curN*1e-3f;
            ray.Direction = dir;
            ray.TMin = 1e-3f;
            ray.TMax = 1e8f;
            uint dummy = 0;
            TraceRay(gAS, RAY_FLAG_NONE, 0xff, 2, 0, 2, ray, dummy);
            float4 b0 = gBounce[p*3 + 0];
            if (b0.w < 0.5f) { L += thr * worldRadiance(); break; }
            thr *= curAlb;
            L += thr * directLighting(b0.xyz, gBounce[p*3 + 1].xyz, gBounce[p*3 + 2].xyz, seed);
            curP = b0.xyz;
            curN = gBounce[p*3 + 1].xyz;
            curAlb = gBounce[p*3 + 2].xyz;
        }
    }
    float3 c = L;
    if (gP2.w > 0.0f) c = min(c, gP2.w);
    gOutput[p] = float4(c, alpha);
}

// ---- closest hit ------------------------------------------------------------
[shader("closesthit")]
void CH_Primary(inout uint payload) {
    uint p = pid();
    float3 o = WorldRayOrigin();
    float3 d = WorldRayDirection();
    float t = RayTMax();
    float3 P = o + d * t;
    uint prim = PrimitiveIndex();
    uint i0 = getIndex(3u*prim + 0u);
    uint i1 = getIndex(3u*prim + 1u);
    uint i2 = getIndex(3u*prim + 2u);
    float3 va = getVertex(i0);
    float3 vb = getVertex(i1);
    float3 vc = getVertex(i2);
    float3 ng = normalize3(cross3(vb - va, vc - va));
    float3 alb = getAlbedo(i0); // C++ always binds a valid albedo buffer
    writeSurface(p, P, ng, alb, gGbuf);
}

[shader("closesthit")]
void CH_Secondary(inout uint payload) {
    uint p = pid();
    float3 o = WorldRayOrigin();
    float3 d = WorldRayDirection();
    float t = RayTMax();
    float3 P = o + d * t;
    uint prim = PrimitiveIndex();
    uint i0 = getIndex(3u*prim + 0u);
    uint i1 = getIndex(3u*prim + 1u);
    uint i2 = getIndex(3u*prim + 2u);
    float3 va = getVertex(i0);
    float3 vb = getVertex(i1);
    float3 vc = getVertex(i2);
    float3 ng = normalize3(cross3(vb - va, vc - va));
    float3 alb = getAlbedo(i0);
    writeSurface(p, P, ng, alb, gBounce);
}

[shader("closesthit")]
void CH_Shadow(inout uint vis) {
    vis = 0; // occluded
}

// ---- miss -------------------------------------------------------------------
[shader("miss")]
void Miss_Primary(inout uint payload) {
    uint p = pid();
    gGbuf[p*3 + 0] = float4(0, 0, 0, 0); // hit flag 0
}

[shader("miss")]
void Miss_Secondary(inout uint payload) {
    uint p = pid();
    gBounce[p*3 + 0] = float4(0, 0, 0, 0); // hit flag 0 -> environment
}

[shader("miss")]
void Miss_Shadow(inout uint vis) {
    vis = 1; // visible
}
