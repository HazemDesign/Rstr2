// Rstr2 Phase 3 - D3D12 / DXR renderer implementation. See renderer.h.

#include "renderer.h"

#include <windows.h>
#include <dxgi1_6.h>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

namespace rstr2 {

namespace {

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

// Camera constant buffer layout must match cbuffer Camera in raytracing.hlsl.
// Each float3 takes a 16-byte (vec4) register; the trailing scalar packs into
// the 4th float of the forward register (offset 60). Total: 4 registers = 64B.
struct CameraCB {
    float origin[4];   // xyz used
    float right[4];    // xyz used
    float up[4];       // xyz used
    float forward[4];  // xyz used; forward[3] = camTanHalfFovY
};
static_assert(sizeof(CameraCB) == 64, "CameraCB must be 64 bytes");

HRESULT serialize_root_signature(const D3D12_ROOT_SIGNATURE_DESC& desc,
                                 ID3DBlob** blob, ID3DBlob** error) {
    return D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, blob, error);
}

// Timestamped diagnostic line to stderr (redirected to bin/Rstr2Core.log).
static void rlogf(const char* fmt, ...) {
    SYSTEMTIME st;
    GetLocalTime(&st);
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[%02u:%02u:%02u.%03u] %s\n",
                 st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, buf);
    std::fflush(stderr);
}

// Default Phase 2 fallback scene: a single triangle + the matching camera.
SceneData make_default_scene() {
    SceneData s;
    s.vertices = {
        -0.7f, -0.5f, 0.0f,
         0.7f, -0.5f, 0.0f,
         0.0f,  0.7f, 0.0f,
    };
    s.indices = {0, 1, 2};
    s.cam_origin[0] = 0.0f; s.cam_origin[1] = 0.6f; s.cam_origin[2] = -2.2f;
    s.cam_right[0] = 1.0f; s.cam_right[1] = 0.0f; s.cam_right[2] = 0.0f;
    s.cam_up[0] = 0.0f; s.cam_up[1] = 1.0f; s.cam_up[2] = 0.0f;
    s.cam_forward[0] = 0.0f; s.cam_forward[1] = 0.0f; s.cam_forward[2] = 1.0f;
    s.cam_tan_half_fov_y = std::tan(0.45f);
    return s;
}

} // namespace

Renderer::Renderer() = default;

Renderer::~Renderer() {
    if (device_ && queue_ && fence_) wait_for_gpu();
    if (fence_event_ != nullptr) CloseHandle(fence_event_);
}

bool Renderer::init(int width, int height, std::string& error) {
    width_ = width;
    height_ = height;
    rlogf("Rstr2Core: init begin\n");

    if (!load_shader_bytecode(error)) return false;
    rlogf("Rstr2Core: init shader ok\n");
    (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    if (!init_dxr(error)) return false;
    rlogf("Rstr2Core: init dxr ok\n");
    if (!create_resources(error)) return false;
    rlogf("Rstr2Core: init resources ok\n");

    // Default fallback scene (hardcoded triangle) so we always have something
    // to render before/without the addon scene bridge.
    scene_ = make_default_scene();
    have_scene_ = true;

    if (!build_scene_accel(error)) return false;
    rlogf("Rstr2Core: init accel ok\n");
    if (!create_root_signatures(error)) return false;
    rlogf("Rstr2Core: init rootsig ok\n");
    if (!create_pipeline(error)) return false;
    rlogf("Rstr2Core: init pipeline ok\n");
    if (!create_sbt(error)) return false;
    rlogf("Rstr2Core: init sbt ok\n");

    return true;
}

bool Renderer::load_shader_bytecode(std::string& error) {
    wchar_t mod[1024] = {};
    DWORD n = GetModuleFileNameW(nullptr, mod, static_cast<DWORD>(_countof(mod)));
    if (n == 0 || n >= _countof(mod)) {
        error = "Rstr2: failed to resolve executable path.";
        return false;
    }
    std::wstring path(mod);
    auto slash = path.find_last_of(L"\\/");
    if (slash != std::wstring::npos) path = path.substr(0, slash + 1);
    path += L"raytracing.cso";

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "Rstr2: raytracing.cso not found next to the executable. "
                "Build the project so shaders/raytracing.hlsl is compiled to raytracing.cso.";
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    if (sz <= 0) { error = "Rstr2: raytracing.cso is empty."; return false; }
    f.seekg(0, std::ios::beg);
    shader_bytecode_.resize(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(shader_bytecode_.data()), sz);
    if (!f) { error = "Rstr2: failed to read raytracing.cso."; return false; }
    return true;
}

bool Renderer::init_dxr(std::string& error) {
    // If a D3D12 debug layer happens to be active (injected by a GPU
    // debugger/profiler such as PIX, Nsight Graphics, RenderDoc, or the
    // Windows "Graphics Tools" feature), disable GPU-Based Validation. GBV
    // performs extremely deep per-call validation that overflows the thread
    // stack during resource/descriptor creation (STATUS_STACK_OVERFLOW).
    // We never enable the debug layer ourselves; this only takes effect if it
    // is already active, and is a harmless no-op otherwise.
    {
        Microsoft::WRL::ComPtr<ID3D12Debug> dbg;
        HRESULT dh = D3D12GetDebugInterface(IID_PPV_ARGS(&dbg));
        if (SUCCEEDED(dh)) {
            Microsoft::WRL::ComPtr<ID3D12Debug3> dbg3;
            if (SUCCEEDED(dbg.As(&dbg3))) {
                dbg3->SetEnableGPUBasedValidation(FALSE);
                dbg3->SetEnableSynchronizedCommandQueueValidation(FALSE);
                rlogf("Rstr2Core: debug layer present, disabled GPU-Based Validation");
            } else {
                rlogf("Rstr2Core: debug layer present (ID3D12Debug only), could not disable GBV");
            }
        } else {
            rlogf("Rstr2Core: no D3D12 debug interface (0x%08X)", static_cast<unsigned>(dh));
        }
    }

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

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
    hr = device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5));
    if (FAILED(hr) || opts5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0) {
        error = "Rstr2: DXR (Raytracing Tier 1.0) is not supported by this GPU/driver.";
        return false;
    }

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
    cmd_list_->Close();

    descriptor_inc_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) { error = "Rstr2: failed to create fence."; return false; }
    fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (fence_event_ == nullptr) { error = "Rstr2: failed to create fence event."; return false; }

    return true;
}

bool Renderer::create_resources(std::string& error) {
    HRESULT hr;
    rlogf("Rstr2Core: create_resources begin\n");

    // Shader-visible CBV_SRV_UAV heap with a single descriptor for the TLAS SRV.
    // (Only the TLAS needs a heap view; everything else is a root descriptor.
    // Creating MANY views into a shader-visible heap crashed the driver stack on
    // the test GPU, so we keep this heap to one descriptor.)
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 1;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    rlogf("Rstr2Core: create_resources heap\n");
    hr = device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_));
    if (FAILED(hr)) { error = "Rstr2: failed to create descriptor heap."; return false; }
    rlogf("Rstr2Core: create_resources heap ok\n");

    const UINT64 pixel_count = static_cast<UINT64>(width_) * static_cast<UINT64>(height_);
    const UINT64 bytes = pixel_count * 16u; // RGBA32F

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
    rlogf("Rstr2Core: created output buffer\n");
    if (FAILED(hr)) { error = "Rstr2: failed to create output buffer."; return false; }

    D3D12_RESOURCE_DESC rb = ob;
    rb.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &rb,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_));
    rlogf("Rstr2Core: created readback buffer");
    if (FAILED(hr)) { error = "Rstr2: failed to create readback buffer."; return false; }

    // Output UAV: bound as a root-descriptor UAV (SetComputeRootUnorderedAccessView)
    // in render_frame, so we deliberately do NOT create a view in this shader-
    // visible heap. (Creating a UAV view into a shader-visible heap triggered a
    // driver stack overflow on the test GPU; the root-descriptor path avoids it.)

    // Camera constant buffer (UPLOAD, 64 bytes).
    D3D12_RESOURCE_DESC cb = {};
    cb.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cb.Width = sizeof(CameraCB);
    cb.Height = 1; cb.DepthOrArraySize = 1; cb.MipLevels = 1;
    cb.SampleDesc.Count = 1;
    cb.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    cb.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kUploadHeap, D3D12_HEAP_FLAG_NONE, &cb,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cam_cbv_));
    rlogf("Rstr2Core: created camera cbv\n");
    if (FAILED(hr)) { error = "Rstr2: failed to create camera constant buffer."; return false; }

    return true;
}

bool Renderer::build_scene_accel(std::string& error) {
    HRESULT hr;
    const UINT64 vcount = scene_.vertices.size() / 3;
    const UINT64 icount = scene_.indices.size();
    if (vcount == 0 || icount == 0 || (icount % 3) != 0) {
        error = "Rstr2: scene has no triangles.";
        return false;
    }
    rlogf("Rstr2Core: build_scene_accel begin (v=%llu i=%llu)\n",
                 (unsigned long long)vcount, (unsigned long long)icount);
    const UINT stride = 12; // float3

    // ---- Vertex buffer (UPLOAD, world space) ----
    D3D12_RESOURCE_DESC vbd = {};
    vbd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    vbd.Width = vcount * stride;
    vbd.Height = 1; vbd.DepthOrArraySize = 1; vbd.MipLevels = 1;
    vbd.SampleDesc.Count = 1;
    vbd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    vbd.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kUploadHeap, D3D12_HEAP_FLAG_NONE, &vbd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vertex_buf_));
    if (FAILED(hr)) { error = "Rstr2: failed to create vertex buffer."; return false; }
    rlogf("Rstr2Core: accel vertex buf\n");
    {
        void* mapped = nullptr;
        vertex_buf_->Map(0, nullptr, &mapped);
        std::memcpy(mapped, scene_.vertices.data(), static_cast<size_t>(vcount) * stride);
        vertex_buf_->Unmap(0, nullptr);
    }

    // ---- Index buffer (UPLOAD, uint32) ----
    D3D12_RESOURCE_DESC ibd = {};
    ibd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ibd.Width = icount * sizeof(uint32_t);
    ibd.Height = 1; ibd.DepthOrArraySize = 1; ibd.MipLevels = 1;
    ibd.SampleDesc.Count = 1;
    ibd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ibd.Flags = D3D12_RESOURCE_FLAG_NONE;
    hr = device_->CreateCommittedResource(
        &kUploadHeap, D3D12_HEAP_FLAG_NONE, &ibd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&index_buf_));
    if (FAILED(hr)) { error = "Rstr2: failed to create index buffer."; return false; }
    rlogf("Rstr2Core: accel index buf\n");
    {
        void* mapped = nullptr;
        index_buf_->Map(0, nullptr, &mapped);
        std::memcpy(mapped, scene_.indices.data(), static_cast<size_t>(icount) * sizeof(uint32_t));
        index_buf_->Unmap(0, nullptr);
    }

    // ---- BLAS ----
    D3D12_RAYTRACING_GEOMETRY_DESC geom = {};
    geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geom.Triangles.VertexBuffer.StartAddress = vertex_buf_->GetGPUVirtualAddress();
    geom.Triangles.VertexBuffer.StrideInBytes = static_cast<UINT>(stride);
    geom.Triangles.VertexCount = static_cast<UINT>(vcount);
    geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geom.Triangles.IndexBuffer = index_buf_->GetGPUVirtualAddress();
    geom.Triangles.IndexCount = static_cast<UINT>(icount);
    geom.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
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
    rlogf("Rstr2Core: accel blas\n");

    D3D12_RESOURCE_DESC scratch_desc = as_desc;
    scratch_desc.Width = blas_pre.ScratchDataSizeInBytes;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &scratch_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&blas_scratch_));
    if (FAILED(hr)) { error = "Rstr2: failed to create BLAS scratch."; return false; }
    rlogf("Rstr2Core: accel blas scratch\n");

    // ---- Instance buffer (identity transform, single instance) ----
    D3D12_RAYTRACING_INSTANCE_DESC inst = {};
    inst.Transform[0][0] = 1.0f; inst.Transform[1][1] = 1.0f; inst.Transform[2][2] = 1.0f;
    inst.InstanceMask = 0xFF;
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
    rlogf("Rstr2Core: accel instance\n");
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
    rlogf("Rstr2Core: accel tlas\n");

    scratch_desc.Width = tlas_pre.ScratchDataSizeInBytes;
    hr = device_->CreateCommittedResource(
        &kDefaultHeap, D3D12_HEAP_FLAG_NONE, &scratch_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&tlas_scratch_));
    if (FAILED(hr)) { error = "Rstr2: failed to create TLAS scratch."; return false; }
    rlogf("Rstr2Core: accel tlas scratch\n");

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
    rlogf("Rstr2Core: accel built\n");

    // Create the TLAS SRV in the shader-visible heap (register t0). Vertices,
    // indices, camera and output are bound as root descriptors in render_frame.
    // If this single CreateShaderResourceView still overflows the driver stack
    // on the test GPU, we will see a crash here; with the 3 GB worker stack it
    // should be fine.
    rlogf("Rstr2Core: accel srv begin\n");
    CD3DX12_CPU_DESCRIPTOR_HANDLE h0(heap_->GetCPUDescriptorHandleForHeapStart());
    D3D12_SHADER_RESOURCE_VIEW_DESC tlas_srv = {};
    tlas_srv.Format = DXGI_FORMAT_UNKNOWN;
    tlas_srv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    tlas_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    tlas_srv.RaytracingAccelerationStructure.Location = tlas_->GetGPUVirtualAddress();
    device_->CreateShaderResourceView(nullptr, &tlas_srv, h0);
    rlogf("Rstr2Core: accel srv done\n");

    return update_camera_cbv(error);
}

bool Renderer::update_camera_cbv(std::string& error) {
    if (!cam_cbv_) { error = "Rstr2: camera constant buffer missing."; return false; }
    CameraCB cb;
    std::memset(&cb, 0, sizeof(cb));
    std::memcpy(cb.origin, scene_.cam_origin, 3 * sizeof(float));
    std::memcpy(cb.right, scene_.cam_right, 3 * sizeof(float));
    std::memcpy(cb.up, scene_.cam_up, 3 * sizeof(float));
    std::memcpy(cb.forward, scene_.cam_forward, 3 * sizeof(float));
    cb.forward[3] = scene_.cam_tan_half_fov_y;
    void* mapped = nullptr;
    cam_cbv_->Map(0, nullptr, &mapped);
    std::memcpy(mapped, &cb, sizeof(cb));
    cam_cbv_->Unmap(0, nullptr);
    return true;
}

bool Renderer::set_scene(const SceneData& scene, std::string& error) {
    scene_ = scene;
    have_scene_ = true;
    // Rebuild geometry + acceleration structures from the new scene.
    return build_scene_accel(error);
}

bool Renderer::create_root_signatures(std::string& error) {
    HRESULT hr;

    // Global root signature:
    //   param 0: descriptor table { SRV t0 (TLAS) } in the shader-visible heap
    //   param 1: root SRV t1 (Vertices)
    //   param 2: root SRV t2 (Indices)
    //   param 3: root CBV b0 (camera)
    //   param 4: root UAV u0 (output)
    // (The TLAS must live in a shader-visible heap; the other resources are root
    // descriptors to minimize heap view creation, which crashed the driver stack
    // on the test GPU when done for many descriptors.)
    D3D12_DESCRIPTOR_RANGE ranges[1] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER params[5] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = ranges;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].Descriptor.ShaderRegister = 1;
    params[1].Descriptor.RegisterSpace = 0;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].Descriptor.ShaderRegister = 2;
    params[2].Descriptor.RegisterSpace = 0;

    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[3].Descriptor.ShaderRegister = 0;
    params[3].Descriptor.RegisterSpace = 0;

    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[4].Descriptor.ShaderRegister = 0;
    params[4].Descriptor.RegisterSpace = 0;

    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 5;
    rs.pParameters = params;
    rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob, err;
    hr = serialize_root_signature(rs, &blob, &err);
    if (FAILED(hr)) { error = "Rstr2: failed to serialize global root signature."; return false; }
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                      IID_PPV_ARGS(&global_rs_));
    if (FAILED(hr)) { error = "Rstr2: failed to create global root signature."; return false; }

    return true;
}

bool Renderer::create_pipeline(std::string& error) {
    rlogf("Rstr2Core: create_pipeline begin\n");
    std::vector<D3D12_STATE_SUBOBJECT> subs;
    subs.reserve(8);

    D3D12_STATE_OBJECT_CONFIG config = {};
    config.Flags = D3D12_STATE_OBJECT_FLAG_NONE;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG;
        s.pDesc = &config;
        subs.push_back(s);
    }

    D3D12_EXPORT_DESC exp[3] = {};
    exp[0].Name = L"raygenMain";
    exp[1].Name = L"missMain";
    exp[2].Name = L"hitMain";
    D3D12_DXIL_LIBRARY_DESC lib = {};
    lib.DXILLibrary.pShaderBytecode = shader_bytecode_.data();
    lib.DXILLibrary.BytecodeLength = static_cast<UINT>(shader_bytecode_.size());
    lib.NumExports = 3;
    lib.pExports = exp;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        s.pDesc = &lib;
        subs.push_back(s);
    }

    D3D12_RAYTRACING_SHADER_CONFIG shader_config = {};
    shader_config.MaxPayloadSizeInBytes = 16;
    shader_config.MaxAttributeSizeInBytes = 8;
    const size_t shader_config_idx = subs.size();
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
        s.pDesc = &shader_config;
        subs.push_back(s);
    }

    D3D12_GLOBAL_ROOT_SIGNATURE global_rs = {};
    global_rs.pGlobalRootSignature = global_rs_.Get();
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
        s.pDesc = &global_rs;
        subs.push_back(s);
    }

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

    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline = {};
    pipeline.MaxTraceRecursionDepth = 1;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
        s.pDesc = &pipeline;
        subs.push_back(s);
    }

    // DXR requires the shader config to be explicitly associated with the
    // shaders that use it. Without this, CreateStateObject fails with
    // E_INVALIDARG because the raygen/miss/hit shaders have no payload or
    // attribute size. (This was the Phase 3 blocker.)
    LPCWSTR shader_config_exports[] = { L"raygenMain", L"missMain", L"HitGroup" };
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shader_config_assoc = {};
    shader_config_assoc.pSubobjectToAssociate = &subs[shader_config_idx];
    shader_config_assoc.NumExports = 3;
    shader_config_assoc.pExports = shader_config_exports;
    {
        D3D12_STATE_SUBOBJECT s = {};
        s.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        s.pDesc = &shader_config_assoc;
        subs.push_back(s);
    }

    D3D12_STATE_OBJECT_DESC so = {};
    so.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    so.NumSubobjects = static_cast<UINT>(subs.size());
    so.pSubobjects = subs.data();

    HRESULT hr = device_->CreateStateObject(&so, IID_PPV_ARGS(&state_object_));
    rlogf("Rstr2Core: CreateStateObject returned 0x%08X\n",
                 static_cast<unsigned>(hr));
    if (FAILED(hr)) {
        error = "Rstr2: failed to create DXR state object (HRESULT 0x" +
                std::to_string(static_cast<unsigned>(hr)) + ").";
        return false;
    }
    hr = state_object_->QueryInterface(IID_PPV_ARGS(&state_props_));
    if (FAILED(hr)) { error = "Rstr2: failed to query ID3D12StateObjectProperties."; return false; }
    return true;
}

bool Renderer::create_sbt(std::string& error) {
    rlogf("Rstr2Core: create_sbt begin\n");
    const UINT idSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT align = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    sbt_raygen_offset_ = 0;
    sbt_miss_offset_ = align;
    sbt_hit_offset_ = align * 2;
    sbt_miss_stride_ = idSize;
    sbt_hit_stride_ = idSize;
    const UINT64 sbt_size = sbt_hit_offset_ + sbt_hit_stride_;

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
    rlogf("Rstr2Core: sbt buffer\n");

    unsigned char* mapped = nullptr;
    sbt_->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    std::memset(mapped, 0, static_cast<size_t>(sbt_size));

    auto copy_id = [&](unsigned char* dst, LPCWSTR name) {
        void* id = state_props_->GetShaderIdentifier(name);
        if (id) std::memcpy(dst, id, idSize);
    };
    copy_id(mapped + sbt_raygen_offset_, L"raygenMain");
    copy_id(mapped + sbt_miss_offset_, L"missMain");
    // The hit-group TABLE must reference the hit-group export ("HitGroup"),
    // not the closest-hit shader ("hitMain") it imports.
    copy_id(mapped + sbt_hit_offset_, L"HitGroup");

    sbt_->Unmap(0, nullptr);
    return true;
}

bool Renderer::render_frame(float* out_pixels, std::string& error) {
    HRESULT hr = cmd_list_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) { error = "Rstr2: command list reset failed."; return false; }

    cmd_list_->SetPipelineState1(state_object_.Get());
    cmd_list_->SetComputeRootSignature(global_rs_.Get());
    // TLAS SRV (t0) via the shader-visible heap table; the rest are root descriptors.
    ID3D12DescriptorHeap* heaps[] = { heap_.Get() };
    cmd_list_->SetDescriptorHeaps(1, heaps);
    cmd_list_->SetComputeRootDescriptorTable(0, heap_->GetGPUDescriptorHandleForHeapStart());
    cmd_list_->SetComputeRootShaderResourceView(1, vertex_buf_->GetGPUVirtualAddress());
    cmd_list_->SetComputeRootShaderResourceView(2, index_buf_->GetGPUVirtualAddress());
    cmd_list_->SetComputeRootConstantBufferView(3, cam_cbv_->GetGPUVirtualAddress());
    cmd_list_->SetComputeRootUnorderedAccessView(4, output_->GetGPUVirtualAddress());

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
