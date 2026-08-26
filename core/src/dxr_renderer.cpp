// Rstr2 DXR 1.1 renderer implementation. See dxr_renderer.h.
//
// Drops in place of the OptiX Renderer: same interface, same SceneData
// contract, same output (RGBA32F via shared_mem::SharedMem::publish_frame).
// Cross-vendor: runs on any DXR 1.1-capable GPU.

#include "dxr_renderer.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

namespace rstr2 {

namespace {

static const unsigned int kMaxBounces = 1u;   // one indirect GI bounce
static const float kTaaClamp = 10.0f;         // firefly clamp, OptiX parity

static void rlogf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "Rstr2Dxr: %s\n", buf);
    fflush(stderr);
}

#define DX_CHECK(call)                                                       \
    do {                                                                     \
        HRESULT _hr = (call);                                                \
        if (FAILED(_hr)) {                                                   \
            rlogf("DXR error 0x%08x at %s:%d in %s", (unsigned)_hr,          \
                  __FILE__, __LINE__, #call);                                \
            error = "Rstr2Dxr: D3D12/DXR call failed (see log).";            \
            return false;                                                    \
        }                                                                    \
    } while (0)

static const unsigned int kThreadStack = 3u * 1024u * 1024u * 1024u; // 3 GB

// Constant buffer (mirrors the HLSL SceneCb cbuffer exactly: 9 * float4).
struct SceneCbData {
    float camOrigin[4];
    float camRight[4];
    float camUp[4];
    float camForward[4];
    float p0[4]; // tanHalfFovY, camShiftX, camShiftY, exposure
    float p1[4]; // width, height, lightCount, worldR
    float p2[4]; // worldG, worldB, worldStrength, taaClamp
    float p3[4]; // frameIndex, jitterX, jitterY, maxBounces
    float p4[4]; // filmTransparent
    float pad[28]; // round the whole CB to 256 bytes
};
static_assert(sizeof(SceneCbData) == 256, "SceneCbData must be 256 bytes");

static std::vector<uint8_t> load_file(const std::wstring& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    if (sz <= 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> d(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(d.data()), sz);
    return d;
}

} // namespace

// ---- tiny CD3DX12-style helpers (d3dx12.h not pulled in) -------------------
static D3D12_RESOURCE_DESC make_buf(UINT64 size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}
static D3D12_RESOURCE_BARRIER uav_barrier(ID3D12Resource* r) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = r;
    return b;
}

struct DxrRenderer::Impl {
    ComPtr<ID3D12Device5> device;
    ComPtr<ID3D12CommandQueue> cmd_queue;
    ComPtr<ID3D12CommandAllocator> cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList4> cmd_list;
    ComPtr<ID3D12Fence> fence;
    uint64_t fence_val = 0;
    HANDLE fence_evt = nullptr;

    ComPtr<ID3D12DescriptorHeap> heap;
    UINT heap_incr = 0;

    ComPtr<ID3D12RootSignature> global_rs;   // descriptors (b0, t0..t4, u0..u2)
    ComPtr<ID3D12RootSignature> empty_local_rs;
    ComPtr<ID3D12StateObject> pso;

    ComPtr<ID3D12Resource> gbuf, bounce, output, output_readback;
    ComPtr<ID3D12Resource> vertices, indices, albedos, lights;
    ComPtr<ID3D12Resource> cb_gpu;
    ComPtr<ID3D12Resource> blas, blas_scratch, tlas, tlas_scratch, instances;
    ComPtr<ID3D12Resource> shader_table;

    size_t vbytes = 0, ibytes = 0, lbytes = 0, abytes = 0;

    std::vector<float> accum;    // CPU EMA (raw HDR, no exposure)
    uint32_t frame_index = 0;
    bool scene_dirty = true;
    bool inited = false;

    // ---- helpers ----------------------------------------------------------
    void wait_gpu() {
        if (!cmd_queue || !fence) return;
        fence_val++;
        cmd_queue->Signal(fence.Get(), fence_val);
        if (fence->GetCompletedValue() < fence_val) {
            fence->SetEventOnCompletion(fence_val, fence_evt);
            WaitForSingleObject(fence_evt, INFINITE);
        }
    }

    bool create_device(std::string& error) {
        ComPtr<IDXGIFactory6> factory;
        DX_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_12_0;
        bool found = false;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d;
            adapter->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), fl, IID_PPV_ARGS(&device)))) {
                D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
                if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
                    o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1) {
                    found = true;
                    rlogf("Rstr2Dxr: adapter '%S' (RT tier %u)", d.Description, (uint32_t)o5.RaytracingTier);
                    break;
                }
                device.Reset();
            }
        }
        if (!found) {
            error = "Rstr2Dxr: no DXR 1.1-capable GPU/adapter found.";
            return false;
        }
        return true;
    }

    bool create_queue_and_heap(std::string& error) {
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        DX_CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&cmd_queue)));

        DX_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&cmd_alloc)));
        DX_CHECK(device->CreateCommandList4(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, nullptr, IID_PPV_ARGS(&cmd_list)));

        DX_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        fence_evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 9;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        DX_CHECK(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        heap_incr = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return true;
    }

    bool create_root_signatures(std::string& error) {
        // Empty local root signature shared by all shaders in the pipeline.
        D3D12_ROOT_SIGNATURE_DESC ers = {};
        ers.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
        ComPtr<ID3DBlob> blob, err_blob;
        if (FAILED(D3D12SerializeRootSignature(&ers, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err_blob))) {
            error = "Rstr2Dxr: serialize empty local root sig failed.";
            return false;
        }
        DX_CHECK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&empty_local_rs)));

        // Global root signature: one descriptor table covering
        //   CBV b0 | SRV t0..t4 | UAV u0..u2
        D3D12_DESCRIPTOR_RANGE ranges[3] = {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        ranges[0].NumDescriptors = 1;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[1].NumDescriptors = 5; // t0..t4
        ranges[1].BaseShaderRegister = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = 1;
        ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[2].NumDescriptors = 3; // u0..u2
        ranges[2].BaseShaderRegister = 0;
        ranges[2].OffsetInDescriptorsFromTableStart = 6;

        D3D12_ROOT_PARAMETER rp = {};
        rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rp.DescriptorTable.NumDescriptorRanges = 3;
        rp.DescriptorTable.pDescriptorRanges = ranges;

        D3D12_ROOT_SIGNATURE_DESC rs = {};
        rs.NumParameters = 1;
        rs.pParameters = &rp;
        rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        ComPtr<ID3DBlob> blob2, err_blob2;
        if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob2, &err_blob2))) {
            error = "Rstr2Dxr: serialize global root sig failed.";
            return false;
        }
        DX_CHECK(device->CreateRootSignature(0, blob2->GetBufferPointer(), blob2->GetBufferSize(), IID_PPV_ARGS(&global_rs)));
        return true;
    }

    bool create_pipeline(std::string& error) {
        // Locate the precompiled DXIL next to the executable.
        wchar_t mod[1024] = {};
        DWORD n = GetModuleFileNameW(nullptr, mod, (DWORD)_countof(mod));
        if (n == 0 || n >= _countof(mod)) { error = "Rstr2Dxr: resolve exe path failed."; return false; }
        std::wstring path(mod);
        auto slash = path.find_last_of(L"\\/");
        if (slash != std::wstring::npos) path = path.substr(0, slash + 1);
        path += L"dxr_shade.cso";
        std::vector<uint8_t> dxil = load_file(path);
        if (dxil.empty()) {
            error = "Rstr2Dxr: dxr_shade.cso not found next to the executable. "
                    "Build it (dxc) from core/shaders/dxr_shade.hlsl (see CMake).";
            return false;
        }

        // Subobjects (reserve so pointers captured by associations stay valid).
        std::vector<D3D12_STATE_SUBOBJECT> subs;
        subs.reserve(16);

        D3D12_DXIL_LIBRARY_DESC lib_desc = {};
        D3D12_SHADER_BYTECODE lib_code = { dxil.data(), dxil.size() };
        lib_desc.DXILLibrary = lib_code;
        // Export every function from the library.
        {
            D3D12_STATE_SUBOBJECT s{};
            s.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
            s.pDesc = &lib_desc;
            subs.push_back(s);
        }

        // Hit groups (each references one closest-hit export).
        auto add_hitgroup = [&](const wchar_t* name, const wchar_t* ch) {
            auto* hg = new D3D12_HIT_GROUP_DESC{};
            hg->HitGroupExport = name;
            hg->ClosestHitShaderImport = ch;
            hg->Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
            D3D12_STATE_SUBOBJECT s{};
            s.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
            s.pDesc = hg;
            subs.push_back(s);
        };
        add_hitgroup(L"HG_Primary", L"CH_Primary");
        add_hitgroup(L"HG_Shadow", L"CH_Shadow");
        add_hitgroup(L"HG_Secondary", L"CH_Secondary");

        // Raytracing shader config.
        auto* rsc = new D3D12_RAYTRACING_SHADER_CONFIG{};
        rsc->MaxPayloadSizeInBytes = 4;     // uint (shadow vis / dummy)
        rsc->MaxAttributeSizeInBytes = 8;   // triangle barycentrics
        {
            D3D12_STATE_SUBOBJECT s{};
            s.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
            s.pDesc = rsc;
            subs.push_back(s);
        }

        // Empty local root signature.
        auto* lrs = new D3D12_LOCAL_ROOT_SIGNATURE{};
        lrs->pLocalRootSignature = empty_local_rs.Get();
        {
            D3D12_STATE_SUBOBJECT s{};
            s.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
            s.pDesc = lrs;
            subs.push_back(s);
        }

        // Associate the empty local RS with all exports.
        const wchar_t* exports[] = { L"RG_Primary", L"RG_Shade", L"CH_Primary",
                                     L"CH_Shadow", L"CH_Secondary",
                                     L"Miss_Primary", L"Miss_Shadow", L"Miss_Secondary" };
        auto* assoc = new D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION{};
        assoc->pSubobjectToAssociate = &subs.back();
        assoc->NumExports = (UINT)_countof(exports);
        assoc->pExports = exports;
        {
            D3D12_STATE_SUBOBJECT s{};
            s.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
            s.pDesc = assoc;
            subs.push_back(s);
        }

        // Pipeline config.
        auto* pc = new D3D12_RAYTRACING_PIPELINE_CONFIG{};
        pc->MaxTraceRecursionDepth = 3; // rg_shade -> GI bounce / shadow
        {
            D3D12_STATE_SUBOBJECT s{};
            s.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
            s.pDesc = pc;
            subs.push_back(s);
        }

        D3D12_STATE_OBJECT_DESC sod = {};
        sod.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
        sod.NumSubobjects = (UINT)subs.size();
        sod.pSubobjects = subs.data();

        DX_CHECK(device->CreateStateObject(&sod, IID_PPV_ARGS(&pso)));
        rlogf("Rstr2Dxr: state object created");
        return true;
    }

    bool create_buffers(int w, int h, std::string& error) {
        const size_t gbuf_bytes = (size_t)w * h * 3u * 16u;
        const size_t out_bytes = (size_t)w * h * 16u;

        auto make_uav = [&](ComPtr<ID3D12Resource>& buf, size_t bytes, UINT numElems) {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = make_buf(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&buf)));
            D3D12_UNORDERED_ACCESS_VIEW_DESC uvd{};
            uvd.Format = DXGI_FORMAT_UNKNOWN; // RWStructuredBuffer<float4>
            uvd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uvd.Buffer.NumElements = numElems;
            uvd.Buffer.StructureByteStride = 16;
            D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += (SIZE_T)(6 * heap_incr); // u0 = gbuf
            if (&buf == &bounce) h.ptr += heap_incr;
            if (&buf == &output) h.ptr += 2 * heap_incr;
            device->CreateUnorderedAccessView(buf.Get(), nullptr, &uvd, h);
            return true;
        };
        make_uav(gbuf, gbuf_bytes, (UINT)(w * h * 3));
        make_uav(bounce, gbuf_bytes, (UINT)(w * h * 3));
        make_uav(output, out_bytes, (UINT)(w * h));

        // Readback for the output (CPU EMA + publish).
        {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC rd = make_buf(out_bytes);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&output_readback)));
        }

        // Constant buffer (UPLOAD so we can map each frame).
        {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC rd = make_buf(256);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cb_gpu)));
            D3D12_CONSTANT_BUFFER_VIEW_DESC cvd{};
            cvd.BufferLocation = cb_gpu->GetGPUVirtualAddress();
            cvd.SizeInBytes = 256;
            D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
            device->CreateConstantBufferView(&cvd, h);
        }

        accum.assign((size_t)w * h * 4, 0.0f);
        return true;
    }

    bool upload_geometry(const SceneData& scene, std::string& error) {
        const size_t vcount = scene.vertices.size() / 3;
        const size_t icount = scene.indices.size();
        if (vcount == 0 || icount == 0 || (icount % 3) != 0) { error = "Rstr2Dxr: scene has no triangles."; return false; }
        vbytes = vcount * 12u;
        ibytes = icount * 4u;
        lbytes = (scene.lights.empty() ? 1u : scene.lights.size()) * 64u;
        abytes = vcount * 12u;

        auto make_srv_raw = [&](ComPtr<ID3D12Resource>& buf, size_t bytes, UINT heapIdx, D3D12_RESOURCE_STATES state) {
            if (buf) buf.Reset();
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = make_buf(bytes);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&buf)));
            return true;
        };
        // Created in COPY_DEST (CopyResource target); promoted to
        // NON_PIXEL_SHADER_RESOURCE after upload so the AS build + shader SRVs
        // can read them.
        make_srv_raw(vertices, vbytes, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        make_srv_raw(indices, ibytes, 2, D3D12_RESOURCE_STATE_COPY_DEST);
        make_srv_raw(albedos, abytes, 3, D3D12_RESOURCE_STATE_COPY_DEST);
        make_srv_raw(lights, lbytes, 4, D3D12_RESOURCE_STATE_COPY_DEST);

        // Default albedos to 0.8 when the scene provides none.
        std::vector<float> alb = scene.albedos;
        if (alb.size() != vcount * 3u) { alb.assign(vcount * 3u, 0.8f); }

        // Lights as 64-byte records.
        std::vector<float> ldata;
        ldata.resize((scene.lights.empty() ? 1u : scene.lights.size()) * 16u, 0.0f);
        for (size_t li = 0; li < scene.lights.size(); ++li) {
            const Light& L = scene.lights[li];
            float* d = &ldata[li * 16u];
            d[0] = L.px; d[1] = L.py; d[2] = L.pz; d[3] = L.type;
            d[4] = L.dx; d[5] = L.dy; d[6] = L.dz; d[7] = L.intensity;
            d[8] = L.cr; d[9] = L.cg; d[10] = L.cb; d[11] = L.size_x;
            d[12] = L.size_y; d[13] = L.ax; d[14] = L.ay; d[15] = L.az;
        }

        auto upload = [&](ID3D12Resource* dst, const void* src, size_t bytes) {
            ComPtr<ID3D12Resource> up;
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC rd = make_buf(bytes);
            device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&up));
            void* m = nullptr;
            up->Map(0, nullptr, &m);
            memcpy(m, src, bytes);
            up->Unmap(0, nullptr);
            cmd_list->CopyResource(dst, up.Get());
        };
        upload(vertices.Get(), scene.vertices.data(), vbytes);
        upload(indices.Get(), scene.indices.data(), ibytes);
        upload(albedos.Get(), alb.data(), abytes);
        upload(lights.Get(), ldata.data(), lbytes);

        // SRV descriptors (raw buffers).
        auto set_raw_srv = [&](ID3D12Resource* r, UINT heapIdx, size_t bytes) {
            D3D12_SHADER_RESOURCE_VIEW_DESC svd{};
            svd.Format = DXGI_FORMAT_R32_TYPELESS;
            svd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            svd.Buffer.NumElements = (UINT)(bytes / 4);
            svd.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            svd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += (SIZE_T)(heapIdx * heap_incr);
            device->CreateShaderResourceView(r, &svd, h);
        };
        set_raw_srv(vertices.Get(), 1, vbytes);
        set_raw_srv(indices.Get(), 2, ibytes);
        set_raw_srv(albedos.Get(), 3, abytes);
        set_raw_srv(lights.Get(), 4, lbytes);

        // Promote geometry buffers to SRV/AS-build readable state.
        D3D12_RESOURCE_BARRIER pb[4] = {};
        for (UINT i = 0; i < 4; ++i) {
            pb[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            pb[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            pb[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        pb[0].Transition.pResource = vertices.Get();
        pb[1].Transition.pResource = indices.Get();
        pb[2].Transition.pResource = albedos.Get();
        pb[3].Transition.pResource = lights.Get();
        cmd_list->ResourceBarrier(4, pb);

        return true;
    }

    bool build_accel(std::string& error) {
        // ---- BLAS ----
        D3D12_RAYTRACING_GEOMETRY_DESC gd = {};
        gd.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        gd.Triangles.VertexBuffer.StartAddress = vertices->GetGPUVirtualAddress();
        gd.Triangles.VertexBuffer.StrideInBytes = 12;
        gd.Triangles.VertexCount = (UINT)(vbytes / 12);
        gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        gd.Triangles.IndexBuffer = indices->GetGPUVirtualAddress();
        gd.Triangles.IndexCount = (UINT)(ibytes / 4);
        gd.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        gd.Triangles.Transform3x4 = 0;
        gd.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE; // no face culling (matches OptiX)

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bi = {};
        bi.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        bi.DescsType = D3D12_ELEMENTS_LAYOUT_ARRAY;
        bi.NumDescs = 1;
        bi.pGeometryDescs = &gd;
        bi.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pbi = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&bi, &pbi);

        if (blas) blas.Reset();
        if (blas_scratch) blas_scratch.Reset();
        {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = make_buf(pbi.ResultDataMaxSizeInBytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&blas)));
            D3D12_RESOURCE_DESC rs = make_buf(pbi.ScratchDataSizeInBytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rs,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&blas_scratch)));
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
        bd.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        bd.Inputs = bi;
        bd.ScratchAccelerationStructureData = blas_scratch->GetGPUVirtualAddress();
        cmd_list->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
        D3D12_RESOURCE_BARRIER bb0 = uav_barrier(blas.Get());
        cmd_list->ResourceBarrier(1, &bb0);

        // ---- TLAS ----
        D3D12_RAYTRACING_INSTANCE_DESC inst = {};
        inst.InstanceMask = 0xFF;
        inst.InstanceContributionToHitGroupIndex = 0;
        inst.Transform[0] = 1; inst.Transform[5] = 1; inst.Transform[10] = 1;
        inst.AccelerationStructure = blas->GetGPUVirtualAddress();
        if (instances) instances.Reset();
        {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC rd = make_buf(sizeof(inst));
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&instances)));
            void* m = nullptr; instances->Map(0, nullptr, &m);
            memcpy(m, &inst, sizeof(inst)); instances->Unmap(0, nullptr);
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tbi = {};
        tbi.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tbi.DescsType = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tbi.NumDescs = 1;
        tbi.InstanceDescs = instances->GetGPUVirtualAddress();
        tbi.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tpbi = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&tbi, &tpbi);
        if (tlas) tlas.Reset();
        if (tlas_scratch) tlas_scratch.Reset();
        {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = make_buf(tpbi.ResultDataMaxSizeInBytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&tlas)));
            D3D12_RESOURCE_DESC rs = make_buf(tpbi.ScratchDataSizeInBytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rs,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&tlas_scratch)));
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tbd = {};
        tbd.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
        tbd.Inputs = tbi;
        tbd.ScratchAccelerationStructureData = tlas_scratch->GetGPUVirtualAddress();
        cmd_list->BuildRaytracingAccelerationStructure(&tbd, 0, nullptr);
        D3D12_RESOURCE_BARRIER bb1 = uav_barrier(tlas.Get());
        cmd_list->ResourceBarrier(1, &bb1);

        // AS SRV (t4) for gAS.
        D3D12_SHADER_RESOURCE_VIEW_DESC asvd{};
        asvd.Format = DXGI_FORMAT_UNKNOWN;
        asvd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        asvd.RaytracingAccelerationStructure.Location = tlas->GetGPUVirtualAddress();
        asvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += (SIZE_T)(5 * heap_incr); // t4
        device->CreateShaderResourceView(nullptr, &asvd, h);
        return true;
    }

    bool build_shader_table(std::string& error) {
        auto get_id = [&](const wchar_t* name, uint8_t* out) -> bool {
            UINT sz = pso->GetShaderIdentifier(name, out, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            if (sz == 0) { error = "Rstr2Dxr: GetShaderIdentifier failed for export."; return false; }
            return true;
        };
        const UINT rec = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES; // 32
        std::vector<uint8_t> st(8 * rec, 0);
        if (!get_id(L"RG_Primary", &st[0 * rec])) return false;
        if (!get_id(L"RG_Shade", &st[1 * rec])) return false;
        if (!get_id(L"HG_Primary", &st[2 * rec])) return false;
        if (!get_id(L"HG_Shadow", &st[3 * rec])) return false;
        if (!get_id(L"HG_Secondary", &st[4 * rec])) return false;
        if (!get_id(L"Miss_Primary", &st[5 * rec])) return false;
        if (!get_id(L"Miss_Shadow", &st[6 * rec])) return false;
        if (!get_id(L"Miss_Secondary", &st[7 * rec])) return false;

        if (shader_table) shader_table.Reset();
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = make_buf(st.size(),
            D3D12_RESOURCE_FLAG_NONE);
        DX_CHECK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&shader_table)));
        void* m = nullptr;
        shader_table->Map(0, nullptr, &m);
        memcpy(m, st.data(), st.size());
        shader_table->Unmap(0, nullptr);
        return true;
    }

    void write_cb(const SceneData& scene, float jitter_x, float jitter_y, float accum_alpha) {
        SceneCbData cb;
        memset(&cb, 0, sizeof(cb));
        cb.camOrigin[0] = scene.cam_origin[0]; cb.camOrigin[1] = scene.cam_origin[1]; cb.camOrigin[2] = scene.cam_origin[2];
        cb.camRight[0] = scene.cam_right[0]; cb.camRight[1] = scene.cam_right[1]; cb.camRight[2] = scene.cam_right[2];
        cb.camUp[0] = scene.cam_up[0]; cb.camUp[1] = scene.cam_up[1]; cb.camUp[2] = scene.cam_up[2];
        cb.camForward[0] = scene.cam_forward[0]; cb.camForward[1] = scene.cam_forward[1]; cb.camForward[2] = scene.cam_forward[2];
        cb.p0[0] = scene.cam_tan_half_fov_y;
        cb.p0[1] = scene.cam_shift[0];
        cb.p0[2] = scene.cam_shift[1];
        cb.p0[3] = (scene.exposure > 0.0f) ? scene.exposure : 1.0f;
        cb.p1[0] = (float)width_;
        cb.p1[1] = (float)height_;
        cb.p1[2] = (float)scene.lights.size();
        cb.p1[3] = scene.world_color[0];
        cb.p2[0] = scene.world_color[1];
        cb.p2[1] = scene.world_color[2];
        cb.p2[2] = scene.world_strength;
        cb.p2[3] = kTaaClamp;
        cb.p3[0] = (float)frame_index;
        cb.p3[1] = jitter_x;
        cb.p3[2] = jitter_y;
        cb.p3[3] = (float)kMaxBounces;
        cb.p4[0] = (scene.flags & kSceneFlagFilmTransparent) ? 1.0f : 0.0f;

        void* m = nullptr;
        cb_gpu->Map(0, nullptr, &m);
        memcpy(m, &cb, sizeof(cb));
        cb_gpu->Unmap(0, nullptr);
    }
};

// ===========================================================================
DxrRenderer::DxrRenderer() = default;
DxrRenderer::~DxrRenderer() { delete impl_; }

bool DxrRenderer::init(int width, int height, std::string& error) {
    width_ = width; height_ = height;
    rlogf("Rstr2Dxr: init begin (%dx%d)\n", width, height);
    impl_ = new Impl();
    if (!impl_->create_device(error)) return false;
    if (!impl_->create_queue_and_heap(error)) return false;
    if (!impl_->create_root_signatures(error)) return false;
    if (!impl_->create_pipeline(error)) return false;
    if (!impl_->create_buffers(width, height, error)) return false;
    if (!impl_->build_shader_table(error)) return false;

    // Default scene so we always have geometry before the addon attaches.
    SceneData def;
    def.vertices = { -0.7f, -0.5f, 0.0f, 0.7f, -0.5f, 0.0f, 0.0f, 0.7f, 0.0f };
    def.indices = { 0, 1, 2 };
    def.cam_origin[0] = 0.0f; def.cam_origin[1] = 0.6f; def.cam_origin[2] = -2.2f;
    def.cam_right[0] = 1.0f; def.cam_up[1] = 1.0f; def.cam_forward[2] = 1.0f;
    def.cam_tan_half_fov_y = std::tan(0.45f);
    if (!set_scene(def, error)) return false;
    impl_->inited = true;
    rlogf("Rstr2Dxr: init complete\n");
    return true;
}

bool DxrRenderer::set_scene(const SceneData& scene, std::string& error) {
    scene_ = scene;
    have_scene_ = true;
    Impl* im = impl_;
    if (!im || !im->inited) { error = "Rstr2Dxr: not initialized."; return false; }

    im->cmd_alloc->Reset();
    im->cmd_list->Reset(im->cmd_alloc.Get(), nullptr);
    if (!im->upload_geometry(scene, error)) return false;
    if (!im->build_accel(error)) return false;
    im->cmd_list->Close();
    ID3D12CommandList* lists[] = { im->cmd_list.Get() };
    im->cmd_queue->ExecuteCommandLists(1, lists);
    im->wait_gpu();
    im->scene_dirty = true;
    rlogf("Rstr2Dxr: set_scene v=%zu i=%zu l=%zu\nexp=%.2f hist=%.1f",
          scene.vertices.size() / 3u, scene.indices.size(), scene.lights.size(),
          scene.exposure, scene.taa_history);
    return true;
}

bool DxrRenderer::resize(int width, int height, std::string& error) {
    Impl* im = impl_;
    if (!im || !im->inited) { error = "Rstr2Dxr: not initialized."; return false; }
    if (width <= 0 || height <= 0 || (width == width_ && height == height_)) return true;
    width_ = width; height_ = height;
    if (im->gbuf) im->gbuf.Reset();
    if (im->bounce) im->bounce.Reset();
    if (im->output) im->output.Reset();
    if (im->output_readback) im->output_readback.Reset();
    if (im->cb_gpu) im->cb_gpu.Reset();
    if (!im->create_buffers(width, height, error)) return false;
    return true;
}

bool DxrRenderer::render_frame(float* out_pixels, std::string& error) {
    Impl* im = impl_;
    if (!im || !im->inited) { error = "Rstr2Dxr: not initialized."; return false; }

    const bool taa_on = (scene_.flags & kSceneFlagTaa) != 0;
    auto frac = [](float x) { return x - std::floor(x); };
    float jitter_x = 0.0f, jitter_y = 0.0f;
    if (taa_on && !im->scene_dirty) {
        jitter_x = frac((float)im->frame_index * 0.7548776662f) - 0.5f;
        jitter_y = frac((float)im->frame_index * 0.5698402909f) - 0.5f;
    }
    float hist = scene_.taa_history;
    if (hist < 1.0f) hist = 1.0f;
    float accum_alpha = (taa_on && !im->scene_dirty) ? (1.0f / hist) : 1.0f;
    float exposure = (scene_.exposure > 0.0f) ? scene_.exposure : 1.0f;

    im->write_cb(scene_, jitter_x, jitter_y, accum_alpha);
    im->scene_dirty = false;

    const UINT rec = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    D3D12_GPU_VIRTUAL_ADDRESS st = im->shader_table->GetGPUVirtualAddress();

    im->cmd_alloc->Reset();
    im->cmd_list->Reset(im->cmd_alloc.Get(), nullptr);
    im->cmd_list->SetDescriptorHeaps(1, im->heap.GetAddressOf());
    im->cmd_list->SetComputeRootSignature(im->global_rs.Get());
    im->cmd_list->SetComputeRootDescriptorTable(0, im->heap->GetGPUDescriptorHandleForHeapStart());

    // Pass 1: primary g-buffer.
    D3D12_DISPATCH_RAYS_DESC dr = {};
    dr.RayGenerationShaderRecord = { st + 0 * rec, rec };
    dr.MissShaderTable = { st + 5 * rec, rec, 3 * rec };
    dr.HitGroupTable = { st + 2 * rec, rec, 3 * rec };
    dr.Width = (UINT)width_; dr.Height = (UINT)height_; dr.Depth = 1;
    im->cmd_list->DispatchRays(&dr);

    // Ensure primary completed before shade reads gbuf.
    D3D12_RESOURCE_BARRIER gbb = uav_barrier(im->gbuf.Get());
    im->cmd_list->ResourceBarrier(1, &gbb);

    // Pass 2: shade.
    dr.RayGenerationShaderRecord = { st + 1 * rec, rec };
    im->cmd_list->DispatchRays(&dr);

    // Copy output -> readback (output must be a valid copy source).
    D3D12_RESOURCE_BARRIER t1{};
    t1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    t1.Transition.pResource = im->output.Get();
    t1.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    t1.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    im->cmd_list->ResourceBarrier(1, &t1);
    im->cmd_list->CopyResource(im->output_readback.Get(), im->output.Get());
    D3D12_RESOURCE_BARRIER t2{};
    t2.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    t2.Transition.pResource = im->output.Get();
    t2.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    t2.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    im->cmd_list->ResourceBarrier(1, &t2);
    im->cmd_list->Close();

    ID3D12CommandList* lists[] = { im->cmd_list.Get() };
    im->cmd_queue->ExecuteCommandLists(1, lists);
    im->wait_gpu();

    // CPU EMA (TAA) + exposure, then copy to out_pixels.
    void* m = nullptr;
    const size_t n = (size_t)width_ * height_ * 4u;
    im->output_readback->Map(0, nullptr, &m);
    const float* frame = static_cast<const float*>(m);
    for (size_t i = 0; i < n; ++i) {
        im->accum[i] += (frame[i] - im->accum[i]) * accum_alpha;
        out_pixels[i] = im->accum[i] * exposure;
    }
    im->output_readback->Unmap(0, nullptr);

    if (im->frame_index < 3u || im->scene_dirty) {
        size_t probe = ((size_t)height_ / 2 * width_ + width_ / 2) * 4u;
        rlogf("Rstr2Dxr: f=%u l=%zu jit=%.3f,%.3f a=%.3f exp=%.2f flg=%u px=%.3f,%.3f,%.3f acc=%.3f,%.3f,%.3f",
              im->frame_index, scene_.lights.size(), jitter_x, jitter_y, accum_alpha,
              exposure, scene_.flags,
              frame[probe], frame[probe + 1], frame[probe + 2],
              im->accum[probe], im->accum[probe + 1], im->accum[probe + 2]);
    }

    im->frame_index++;
    return true;
}

} // namespace rstr2

