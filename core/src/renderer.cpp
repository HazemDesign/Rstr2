// Rstr2 Phase 2 - D3D12 / DXR renderer implementation. See renderer.h.

#include "renderer.h"
#include "raytracing_cso.h"

#include <windows.h>
#include <dxgi1_6.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>

namespace rstr2 {

namespace {

// Vertex layout for the hardcoded triangle.
struct TriVertex { float x, y, z; };

// Minimal stand-ins for the d3dx12.h helpers we are not allowed to use.
struct CD3DX12_CPU_DESCRIPTOR_HANDLE : public D3D12_CPU_DESCRIPTOR_HANDLE {
    CD3DX12_CPU_DESCRIPTOR_HANDLE() = default;
    CD3DX12_CPU_DESCRIPTOR_HANDLE(const D3D12_CPU_DESCRIPTOR_HANDLE& o) { *this = o; }
    void Offset(UINT count, UINT inc) { ptr += static_cast<SIZE_T>(count) * inc; }
};

const D3D12_HEAP_PROPERTIES kDefaultHeap = []() {
    D3D12_HEAP_PROPERTIES h = {};
    h.Type = D3D12_HEAP_TYPE_DEFAULT;
    return h;
}();
const D3D12_HEAP_PROPERTIES kUploadHeap = []() {
    D3D12_HEAP_PROPERTIES h = {};
    h.Type = D3D12_HEAP_TYPE_UPLOAD;
    return h;
}();

HRESULT serialize_root_signature(const D3D12_ROOT_SIGNATURE_DESC& desc,
                                 ID3DBlob** blob, ID3DBlob** error) {
    return D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, blob, error);
}

} // namespace

Renderer::Renderer() = default;

Renderer::~Renderer() {
    // Ensure GPU is idle before releasing resources.
    if (device_ && queue_ && fence_) {
        wait_for_gpu();
    }
    if (fence_event_ != nullptr) CloseHandle(fence_event_);
}

bool Renderer::init(int width, int height, std::string& error) {
    width_ = width;
    height_ = height;

    if (g_raytracing_cso_size == 0) {
        error = "Rstr2: raytracing shader bytecode is not embedded "
                "(raytracing_cso.h is a stub). Build with dxc available so that "
                "shaders/raytracing.hlsl is compiled into core/src/raytracing_cso.h.";
        return false;
    }

    (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED); // not fatal if already init

    if (!init_dxr(error)) return false;
    if (!create_resources(error)) return false;
    if (!build_acceleration_structures(error)) return false;
    if (!create_root_signatures(error)) return false;
    if (!create_pipeline(error)) return false;
    if (!create_sbt(error)) return false;

    return true;
}

bool Renderer::init_dxr(std::string& error) {
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory_));
    if (FAILED(hr)) {
        error = "Rstr2: failed to create DXGI factory (HRESULT 0x" +
                std::to_string(static_cast<unsigned>(hr)) + ").";
        return false;
    }

    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    bool found = false;
    for (UINT i = 0; factory_->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue; // skip WARP
        hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_));
        if (SUCCEEDED(hr)) { found = true; break; }
        adapter.Reset();
    }
    if (!found) {
        error = "Rstr2: no D3D12-capable GPU adapter found (or D3D12CreateDevice failed).";
        return false;
    }

    // Raytracing support?
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
    hr = device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5));
    if (FAILED(hr) || opts5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0) {
        error = "Rstr2: DXR (Raytracing Tier 1.0) is not supported by this GPU/driver.";
        return false;
    }

    // GRAPHICS (DIRECT) command queue.
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_));
    if (FAILED(hr)) { error = "Rstr2: failed to create command queue."; return false; }

    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_));
    if (FAILED(hr)) { error = "Rstr2: failed to create command allocator."; return false; }

    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(),
                                    nullptr, IID_PPV_ARGS(&cmd_list_));
    if (FAILED(hr)) { error = "Rstr2: failed to create command list."; return false; }
    cmd_list_->Close(); // created in recording state; reset before use.

    descriptor_inc_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) { error = "Rstr2: failed to create fence."; return false; }
    fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (fence_event_ == nullptr) { error = "Rstr2: failed to create fence event."; return false; }

    return true;
}

bool Renderer::create_resources(std::string& error) {
    HRESULT hr;

    // Shader-visible CBV_SRV_UAV heap: [0]=TLAS SRV, [1]=output UAV.
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 2;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_));
    if (FAILED(hr)) { error = "Rstr2: failed to create descriptor heap."; return false; }

    const UINT64 pixel_count = static_cast<UINT64>(width_) * static_cast<UINT64>(height_);
    const UINT64 bytes = pixel_count * 16u; // RGBA32F

    // Output RW buffer (DEFAULT, UNORDERED_ACCESS).
    D3D12_RESOURCE_DESC ob = {};
    ob.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ob.Width = bytes;
    ob.Height = 1; ob.DepthOrArraySize = 1; ob.MipLevels = 1;
    ob.SampleDesc.Count = 1;
    ob.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ob.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &ob,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&output_));
    if (FAILED(hr)) { error = "Rstr2: failed to create output buffer."; return false; }

    // Readback buffer (COPY_DEST initially).
    D3D12_RESOURCE_DESC rb = ob;
    rb.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &rb,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_));
    if (FAILED(hr)) { error = "Rstr2: failed to create readback buffer."; return false; }

    // Vertex buffer (UPLOAD) with the 3 hardcoded triangle vertices.
    TriVertex verts[3] = {
        {-0.7f, -0.5f, 0.0f},
        { 0.7f, -0.5f, 0.0f},
        { 0.0f,  0.7f, 0.0f},
    };
    const UINT64 vsize = sizeof(verts);
    D3D12_RESOURCE_DESC vb = {};
    vb.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    vb.Width = vsize;
    vb.Height = 1; vb.DepthOrArraySize = 1; vb.MipLevels = 1;
    vb.SampleDesc.Count = 1;
    vb.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    vb.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kUploadHeap, D3D12_HEAP_FLAG_NONE, &vb,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vertex_buf_));
    if (FAILED(hr)) { error = "Rstr2: failed to create vertex buffer."; return false; }
    {
        void* mapped = nullptr;
        vertex_buf_->Map(0, nullptr, &mapped);
        std::memcpy(mapped, verts, sizeof(verts));
        vertex_buf_->Unmap(0, nullptr);
    }

    return true;
}

bool Renderer::build_acceleration_structures(std::string& error) {
    HRESULT hr;
    const UINT64 vstride = sizeof(TriVertex);

    // ---- BLAS ----
    D3D12_RAYTRACING_GEOMETRY_DESC geom = {};
    geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geom.Triangles.VertexBuffer.StartAddress = vertex_buf_->GetGPUVirtualAddress();
    geom.Triangles.VertexBuffer.StrideInBytes = static_cast<UINT>(vstride);
    geom.Triangles.VertexCount = 3;
    geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geom.Triangles.IndexFormat = DXGI_FORMAT_UNKNOWN;
    geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs = {};
    blas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    blas_inputs.NumDescs = 1;
    blas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blas_inputs.pGeometryDescs = &geom;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_pre = {};
    device_->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &blas_pre);

    D3D12_RESOURCE_DESC as_desc = {};
    as_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    as_desc.Height = 1; as_desc.DepthOrArraySize = 1; as_desc.MipLevels = 1;
    as_desc.SampleDesc.Count = 1;
    as_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    as_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    as_desc.Width = blas_pre.ResultDataMaxSizeInBytes;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &as_desc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&blas_));
    if (FAILED(hr)) { error = "Rstr2: failed to create BLAS buffer."; return false; }

    D3D12_RESOURCE_DESC scratch_desc = as_desc;
    scratch_desc.Width = blas_pre.ScratchDataSizeInBytes;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &scratch_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&blas_scratch_));
    if (FAILED(hr)) { error = "Rstr2: failed to create BLAS scratch."; return false; }

    // ---- Instance buffer (UPLOAD) ----
    D3D12_RAYTRACING_INSTANCE_DESC inst = {};
    inst.Transform[0][0] = 1.0f; inst.Transform[1][1] = 1.0f; inst.Transform[2][2] = 1.0f;
    inst.InstanceMask = 1;
    inst.InstanceContributionToHitGroupIndex = 0;
    inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    inst.AccelerationStructure = blas_->GetGPUVirtualAddress();
    inst.InstanceID = 0;

    D3D12_RESOURCE_DESC ib = {};
    ib.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ib.Width = sizeof(inst);
    ib.Height = 1; ib.DepthOrArraySize = 1; ib.MipLevels = 1;
    ib.SampleDesc.Count = 1;
    ib.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ib.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kUploadHeap, D3D12_HEAP_FLAG_NONE, &ib,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&instance_buf_));
    if (FAILED(hr)) { error = "Rstr2: failed to create instance buffer."; return false; }
    {
        void* mapped = nullptr;
        instance_buf_->Map(0, nullptr, &mapped);
        std::memcpy(mapped, &inst, sizeof(inst));
        instance_buf_->Unmap(0, nullptr);
    }

    // ---- TLAS ----
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs = {};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    tlas_inputs.NumDescs = 1;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.InstanceDescs = instance_buf_->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_pre = {};
    device_->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &tlas_pre);

    as_desc.Width = tlas_pre.ResultDataMaxSizeInBytes;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &as_desc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&tlas_));
    if (FAILED(hr)) { error = "Rstr2: failed to create TLAS buffer."; return false; }

    scratch_desc.Width = tlas_pre.ScratchDataSizeInBytes;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &scratch_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&tlas_scratch_));
    if (FAILED(hr)) { error = "Rstr2: failed to create TLAS scratch."; return false; }

    // ---- Record + execute the two builds ----
    hr = cmd_list_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) { error = "Rstr2: command list reset failed."; return false; }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blas_build = {};
    blas_build.DestAccelerationStructureData = blas_->GetGPUVirtualAddress();
    blas_build.Inputs = blas_inputs;
    blas_build.ScratchAccelerationStructureData = blas_scratch_->GetGPUVirtualAddress();
    cmd_list_->BuildRaytracingAccelerationStructure(&blas_build, 0, nullptr);

    D3D12_RESOURCE_BARRIER blas_barrier = {};
    blas_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    blas_barrier.UAV.pResource = blas_.Get();
    cmd_list_->ResourceBarrier(1, &blas_barrier);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_build = {};
    tlas_build.DestAccelerationStructureData = tlas_->GetGPUVirtualAddress();
    tlas_build.Inputs = tlas_inputs;
    tlas_build.ScratchAccelerationStructureData = tlas_scratch_->GetGPUVirtualAddress();
    cmd_list_->BuildRaytracingAccelerationStructure(&tlas_build, 0, nullptr);

    D3D12_RESOURCE_BARRIER tlas_barrier = {};
    tlas_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    tlas_barrier.UAV.pResource = tlas_.Get();
    cmd_list_->ResourceBarrier(1, &tlas_barrier);

    hr = cmd_list_->Close();
    if (FAILED(hr)) { error = "Rstr2: command list close failed."; return false; }
    ID3D12CommandList* lists[] = { cmd_list_.Get() };
    queue_->ExecuteCommandLists(1, lists);
    wait_for_gpu();

    // ---- TLAS SRV (heap index 0) ----
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.RaytracingAccelerationStructure.Location = tlas_->GetGPUVirtualAddress();
    CD3DX12_CPU_DESCRIPTOR_HANDLE h0(heap_->GetCPUDescriptorHandleForHeapStart());
    device_->CreateShaderResourceView(nullptr, &srv, h0);

    // ---- Output UAV (heap index 1) ----
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = static_cast<UINT>(width_ * height_);
    uav.Buffer.StructureByteStride = 0;
    CD3DX12_CPU_DESCRIPTOR_HANDLE h1(h0);
    h1.Offset(1, descriptor_inc_);
    device_->CreateUnorderedAccessView(output_.Get(), nullptr, &uav, h1);

    return true;
}

bool Renderer::create_root_signatures(std::string& error) {
    HRESULT hr;

    // Global root signature: descriptor table { SRV t0, UAV u0 }.
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

    D3D12_ROOT_PARAMETER gparam = {};
    gparam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    gparam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    gparam.DescriptorTable.NumDescriptorRanges = 2;
    gparam.DescriptorTable.pDescriptorRanges = ranges;

    D3D12_ROOT_SIGNATURE_DESC grs = {};
    grs.NumParameters = 1;
    grs.pParameters = &gparam;
    grs.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob, err;
    hr = serialize_root_signature(grs, &blob, &err);
    if (FAILED(hr)) { error = "Rstr2: failed to serialize global root signature."; return false; }
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                      IID_PPV_ARGS(&global_rs_));
    if (FAILED(hr)) { error = "Rstr2: failed to create global root signature."; return false; }

    // Local root signature (hit): root SRV t0, space1 (vertex buffer).
    D3D12_ROOT_PARAMETER lparam = {};
    lparam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    lparam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    lparam.SRV.ShaderRegister = 0;
    lparam.SRV.RegisterSpace = 1;
    lparam.SRV.Flags = D3D12_ROOT_SRV_FLAG_NONE;

    D3D12_ROOT_SIGNATURE_DESC lrs = {};
    lrs.NumParameters = 1;
    lrs.pParameters = &lparam;
    lrs.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    hr = serialize_root_signature(lrs, &blob, &err);
    if (FAILED(hr)) { error = "Rstr2: failed to serialize local root signature."; return false; }
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                      IID_PPV_ARGS(&local_rs_));
    if (FAILED(hr)) { error = "Rstr2: failed to create local root signature."; return false; }

    return true;
}

bool Renderer::create_pipeline(std::string& error) {
    std::vector<D3D12_STATE_SUBOBJECT> subs;
    subs.reserve(8); // reserve so captured pointers stay valid

    // 1. State object config.
    D3D12_STATE_OBJECT_CONFIG config = {};
    config.Flags = D3D12_STATE_OBJECT_FLAG_NONE;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG;
        s.pDesc = &config;
        subs.push_back(s);
    }

    // 2. DXIL library with three exports.
    D3D12_EXPORT_DESC exp[3] = {};
    exp[0].Name = L"raygenMain";
    exp[1].Name = L"missMain";
    exp[2].Name = L"hitMain";
    D3D12_DXIL_LIBRARY_DESC lib = {};
    lib.DXILLibrary.pShaderBytecode = g_raytracing_cso;
    lib.DXILLibrary.SizeInBytes = g_raytracing_cso_size;
    lib.NumExports = 3;
    lib.pExports = exp;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        s.pDesc = &lib;
        subs.push_back(s);
    }

    // 3. Shader config (payload 16, attr 8).
    D3D12_RAYTRACING_SHADER_CONFIG shader_config = {};
    shader_config.MaxPayloadSizeInBytes = 16;
    shader_config.MaxAttributeSizeInBytes = 8;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
        s.pDesc = &shader_config;
        subs.push_back(s);
    }

    // 4. Global root signature.
    D3D12_GLOBAL_ROOT_SIGNATURE global_rs = {};
    global_rs.pRootSignature = global_rs_.Get();
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
        s.pDesc = &global_rs;
        subs.push_back(s);
    }

    // 5. Local root signature (kept referenced for the association below).
    D3D12_LOCAL_ROOT_SIGNATURE local_rs = {};
    local_rs.pRootSignature = local_rs_.Get();
    D3D12_STATE_SUBOBJECT local_rs_sub = {};
    local_rs_sub.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    local_rs_sub.pDesc = &local_rs;
    subs.push_back(local_rs_sub);
    D3D12_STATE_SUBOBJECT* local_rs_ptr = &subs.back();

    // 6. Hit group.
    D3D12_HIT_GROUP_DESC hg = {};
    hg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    hg.HitGroupExport = L"HitGroup";
    hg.ClosestHitShaderImport = L"hitMain";
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        s.pDesc = &hg;
        subs.push_back(s);
    }

    // 7. Associate local root signature with the hit group.
    LPCWSTR assoc_exports[] = { L"HitGroup" };
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION assoc = {};
    assoc.pSubobjectToAssociate = local_rs_ptr;
    assoc.NumExports = 1;
    assoc.pExports = assoc_exports;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        s.pDesc = &assoc;
        subs.push_back(s);
    }

    // 8. Pipeline config (max recursion depth 1).
    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline = {};
    pipeline.MaxTraceRecursionDepth = 1;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
        s.pDesc = &pipeline;
        subs.push_back(s);
    }

    D3D12_STATE_OBJECT_DESC so = {};
    so.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    so.NumSubobjects = static_cast<UINT>(subs.size());
    so.pSubobjects = subs.data();

    HRESULT hr = device_->CreateStateObject(&so, IID_PPV_ARGS(&state_object_));
    if (FAILED(hr)) {
        error = "Rstr2: failed to create DXR state object (HRESULT 0x" +
                std::to_string(static_cast<unsigned>(hr)) + ").";
        return false;
    }
    hr = state_object_->QueryInterface(IID_PPV_ARGS(&state_props_));
    if (FAILED(hr)) {
        error = "Rstr2: failed to query ID3D12StateObjectProperties.";
        return false;
    }
    return true;
}

bool Renderer::create_sbt(std::string& error) {
    const UINT idSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;   // 32
    const UINT align = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT; // 64
    sbt_raygen_offset_ = 0;
    sbt_miss_offset_ = align;        // 64
    sbt_hit_offset_ = align * 2;     // 128
    sbt_miss_stride_ = idSize;       // 32
    sbt_hit_stride_ = 64;            // id(32) + localArg(8) -> aligned to 32
    const UINT64 sbt_size = sbt_hit_offset_ + sbt_hit_stride_; // 192

    D3D12_RESOURCE_DESC sd = {};
    sd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    sd.Width = sbt_size;
    sd.Height = 1; sd.DepthOrArraySize = 1; sd.MipLevels = 1;
    sd.SampleDesc.Count = 1;
    sd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    sd.Flags = D3D12_RESOURCE_FLAG_NONE;
    HRESULT hr = device_->CreateCommittedResource(
        &kUploadHeap, D3D12_HEAP_FLAG_NONE, &sd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&sbt_));
    if (FAILED(hr)) { error = "Rstr2: failed to create SBT buffer."; return false; }

    unsigned char* mapped = nullptr;
    sbt_->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    std::memset(mapped, 0, static_cast<size_t>(sbt_size));

    auto copy_id = [&](unsigned char* dst, LPCWSTR name) {
        void* id = state_props_->GetShaderIdentifier(name);
        if (id) std::memcpy(dst, id, idSize);
    };
    copy_id(mapped + sbt_raygen_offset_, L"raygenMain");
    copy_id(mapped + sbt_miss_offset_, L"missMain");
    copy_id(mapped + sbt_hit_offset_, L"hitMain");

    // Local root argument for the hit group: GPU VA of the vertex buffer (8 bytes).
    const UINT64 vtx_va = vertex_buf_->GetGPUVirtualAddress();
    std::memcpy(mapped + sbt_hit_offset_ + idSize, &vtx_va, sizeof(vtx_va));

    sbt_->Unmap(0, nullptr);
    return true;
}

bool Renderer::render_frame(float* out_pixels, std::string& error) {
    HRESULT hr = cmd_list_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) { error = "Rstr2: command list reset failed."; return false; }

    ID3D12DescriptorHeap* heaps[] = { heap_.Get() };
    cmd_list_->SetDescriptorHeaps(1, heaps);
    cmd_list_->SetPipelineState1(state_object_.Get());
    cmd_list_->SetComputeRootSignature(global_rs_.Get());
    cmd_list_->SetComputeRootDescriptorTable(0, heap_->GetGPUDescriptorHandleForHeapStart());

    D3D12_DISPATCH_RAYS_DESC dr = {};
    dr.RayGenerationShaderRecord.StartAddress = sbt_->GetGPUVirtualAddress() + sbt_raygen_offset_;
    dr.RayGenerationShaderRecord.SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    dr.MissShaderTable.StartAddress = sbt_->GetGPUVirtualAddress() + sbt_miss_offset_;
    dr.MissShaderTable.SizeInBytes = sbt_miss_stride_;
    dr.HitGroupTable.StartAddress = sbt_->GetGPUVirtualAddress() + sbt_hit_offset_;
    dr.HitGroupTable.SizeInBytes = sbt_hit_stride_;
    dr.Width = static_cast<UINT>(width_);
    dr.Height = static_cast<UINT>(height_);
    dr.Depth = 1;
    cmd_list_->DispatchRays(&dr);

    // Output buffer: UNORDERED_ACCESS -> COPY_SOURCE.
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = output_.Get();
    b.Transition.Subresource = 0;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmd_list_->ResourceBarrier(1, &b);

    const UINT64 bytes = static_cast<UINT64>(width_) * static_cast<UINT64>(height_) * 16u;
    cmd_list_->CopyBufferRegion(readback_.Get(), 0, output_.Get(), 0, bytes);

    hr = cmd_list_->Close();
    if (FAILED(hr)) { error = "Rstr2: command list close failed."; return false; }
    ID3D12CommandList* lists[] = { cmd_list_.Get() };
    queue_->ExecuteCommandLists(1, lists);
    wait_for_gpu();

    void* rb = nullptr;
    readback_->Map(0, nullptr, &rb);
    std::memcpy(out_pixels, rb, static_cast<size_t>(bytes));
    readback_->Unmap(0, nullptr);

    return true;
}

void Renderer::wait_for_gpu() {
    ++fence_value_;
    queue_->Signal(fence_.Get(), fence_value_);
    if (fence_->GetCompletedValue() < fence_value_) {
        fence_->SetEventOnCompletion(fence_value_, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

} // namespace rstr2
