// Rstr2 Phase 3 - D3D12 / DXR renderer.
//
// Ray-traces a world-space triangle mesh supplied by the Blender addon
// (via the scene shared-memory bridge) with a camera also supplied by the
// addon. Falls back to a single hardcoded triangle when no scene is set.
// No d3dx12.h; only the Windows SDK + d3d12/dxgi/dxguid libs.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include "shared_mem.h"

namespace rstr2 {

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Initialize the device, default scene, pipeline and SBT.
    // Returns false on any failure and fills `error`.
    bool init(int width, int height, std::string& error);

    // Render a single frame. `out_pixels` must hold width*height*4 floats.
    bool render_frame(float* out_pixels, std::string& error);

    // Replace the scene (geometry + camera) and rebuild acceleration
    // structures. Safe to call between frames from the render loop.
    bool set_scene(const SceneData& scene, std::string& error);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    bool init_dxr(std::string& error);
    bool create_resources(std::string& error);
    bool build_scene_accel(std::string& error);
    bool update_camera_cbv(std::string& error);
    bool create_root_signatures(std::string& error);
    bool create_pipeline(std::string& error);
    bool create_sbt(std::string& error);
    bool load_shader_bytecode(std::string& error);
    void wait_for_gpu();

    int width_ = 0;
    int height_ = 0;

    SceneData scene_;
    bool have_scene_ = false;

    // Raytracing shader bytecode (loaded at runtime from <exedir>/raytracing.cso).
    std::vector<uint8_t> shader_bytecode_;

    // D3D12 objects.
    Microsoft::WRL::ComPtr<IDXGIFactory1>        factory_;
    Microsoft::WRL::ComPtr<ID3D12Device5>        device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>   queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       output_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       readback_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       cam_cbv_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       vertex_buf_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       index_buf_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       instance_buf_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       blas_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       blas_scratch_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       tlas_;
    Microsoft::WRL::ComPtr<ID3D12Resource>       tlas_scratch_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>  global_rs_;
    Microsoft::WRL::ComPtr<ID3D12StateObject>     state_object_;
    Microsoft::WRL::ComPtr<ID3D12StateObjectProperties> state_props_;
    Microsoft::WRL::ComPtr<ID3D12Resource>        sbt_;
    Microsoft::WRL::ComPtr<ID3D12Fence>          fence_;
    HANDLE fence_event_ = nullptr;

    UINT descriptor_inc_ = 0;
    size_t sbt_raygen_offset_ = 0;
    size_t sbt_miss_offset_ = 0;
    size_t sbt_hit_offset_ = 0;
    size_t sbt_miss_stride_ = 0;
    size_t sbt_hit_stride_ = 0;
    UINT64 fence_value_ = 0;
};

} // namespace rstr2
