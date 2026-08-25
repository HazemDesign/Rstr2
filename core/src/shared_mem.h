// Rstr2 - shared memory bridges (writer: frame; reader: scene).
//
// --- Frame bridge (Phase 2, writer side, core -> addon) -----------------
// Layout (little-endian), total header = 256 bytes:
//   offset 0  : u32 magic        = 0x32525352 ('Rstr' little-endian-ish)
//   offset 4  : u32 version      = 1
//   offset 8  : i32 width
//   offset 12 : i32 height
//   offset 16 : u32 frame_index  (incremented LAST, with release semantics)
//   offset 20 : u32 state        (1 = ready)
//   offset 24 : u64 pixel_offset = 256
//   offset 32 : u64 pixel_size   = width*height*16
//   offset 40 : (padding to 256)
//   offset 256: RGBA32F pixels, row-major, first row = TOP, alpha = 1.
//
// Publish order (writer): memcpy pixels -> release fence -> state=1 ->
// frame_index++ LAST. The reader detects a new frame by watching frame_index.
//
// --- Scene bridge (Phase 3, reader side, addon -> core) ------------------
// A separate mapping "Local\\Rstr2Scene_v1" carries the Blender scene the
// core should ray-trace: world-space triangle soup + camera basis. The core
// polls it; the addon (Python) writes it. Header (256 bytes) at offset 0:
//   u32 magic        = 0x32525353 ('RRS3')
//   u32 version      = 2
//   u32 epoch        (incremented LAST by the writer on each update)
//   u32 ready        (1 = valid)
//   u32 writing      (1 while the writer is mid-update)
//   u32 vertex_count (xyz triples)
//   u32 index_count  (uint32 indices)
//   u32 light_count  (typed lights, 16 floats each - see Light)
//   u32 flags        (bit0 = TAA enabled)
//   float exposure, taa_history
//   float cam_origin[3], cam_right[3], cam_up[3], cam_forward[3]
//   float cam_tan_half_fov_y
//   (padded to 256)
//   offset 256: vertices  (vertex_count * 3 * float32, world space)
//   then      : indices   (index_count * uint32)
//   then      : lights    (light_count * 16 * float32 - see Light layout)
//   then      : albedos   (vertex_count * 3 * float32, linear RGB per vertex)

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "optix_params.h"

namespace rstr2 {

// ----------------------------------------------------------------------
// Scene bridge types
// ----------------------------------------------------------------------
static constexpr uint32_t kSceneMagic = 0x32525353u;
static constexpr uint32_t kSceneVersion = 2;
static constexpr size_t kSceneHeaderSize = 256;
static constexpr size_t kMaxSceneBytes = 64 * 1024 * 1024; // 64 MB safety cap

static constexpr uint32_t kSceneFlagTaa = 1u << 0;
static constexpr uint32_t kSceneFlagFilmTransparent = 1u << 1;

struct SceneData {
    std::vector<float> vertices;       // xyz, world space
    std::vector<uint32_t> indices;     // uint32 triangle indices
    std::vector<Light> lights;         // typed-light pool (RTXDI)
    std::vector<float> albedos;        // rgb triples per vertex (may be empty)
    uint32_t flags = kSceneFlagTaa;
    float exposure = 1.0f;
    float taa_history = 20.0f;
    // World/environment light (uniform-color approximation), packed into the
    // header's reserved area by v2 writers. strength == 0 => no world light.
    float world_color[3] = {0.05f, 0.05f, 0.05f};
    float world_strength = 0.0f;
    // Requested render size (viewport/F12). 0 = keep current.
    uint32_t render_width = 0;
    uint32_t render_height = 0;
    float cam_origin[3] = {0, 0, 0};
    float cam_right[3] = {1, 0, 0};
    float cam_up[3] = {0, 1, 0};
    float cam_forward[3] = {0, 0, 1};
    float cam_tan_half_fov_y = 0.5f;
};

#pragma pack(push, 4)
struct SceneHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t epoch;
    uint32_t ready;
    uint32_t writing;
    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t light_count;
    uint32_t flags;
    float exposure;
    float taa_history;
    float cam_origin[3];
    float cam_right[3];
    float cam_up[3];
    float cam_forward[3];
    float cam_tan_half_fov_y;
    uint8_t reserved[kSceneHeaderSize - (9 * 4 + 15 * 4)];
};
#pragma pack(pop)

static_assert(sizeof(SceneHeader) == kSceneHeaderSize, "SceneHeader must be 256 bytes");


#pragma pack(push, 8)
struct FrameHeader {
    uint32_t magic;
    uint32_t version;
    int32_t  width;
    int32_t  height;
    uint32_t frame_index;
    uint32_t state;
    uint64_t pixel_offset;
    uint64_t pixel_size;
    uint8_t  reserved[256 - 40];
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 256, "FrameHeader must be exactly 256 bytes");
static_assert(offsetof(FrameHeader, pixel_offset) == 24, "pixel_offset field offset");
static_assert(offsetof(FrameHeader, pixel_size) == 32, "pixel_size field offset");

class SharedMem {
public:
    // Create (or fail) a named shared-memory section of `total_size` bytes.
    SharedMem(const std::wstring& name, size_t total_size);
    ~SharedMem();

    SharedMem(const SharedMem&) = delete;
    SharedMem& operator=(const SharedMem&) = delete;

    bool valid() const { return view_ != nullptr; }
    size_t size() const { return size_; }

    // Publish one frame. `pixels` must point at width*height*4 floats.
    void publish_frame(const float* pixels, int width, int height);

private:
    void* view_ = nullptr;
    void* mapping_ = nullptr;
    size_t size_ = 0;
    uint32_t frame_index_ = 0;
};

// ----------------------------------------------------------------------
// Scene bridge (reader side). The addon creates the mapping and writes to
// it; the core opens it and polls for scene updates.
// ----------------------------------------------------------------------
class SceneMem {
public:
    explicit SceneMem(const std::wstring& name);
    ~SceneMem();

    SceneMem(const SceneMem&) = delete;
    SceneMem& operator=(const SceneMem&) = delete;

    // Attach to the mapping (best effort; safe to call repeatedly).
    bool open();
    bool is_open() const { return view_ != nullptr; }
    void close();

    // Copy out a new scene if `epoch` changed and the write was consistent.
    // Returns false (and leaves `out` untouched) otherwise.
    bool read_scene(SceneData& out);

private:
    bool consistent_copy(SceneData& out);

    std::wstring name_;
    void* mapping_ = nullptr;
    void* view_ = nullptr;
    uint32_t last_epoch_ = 0;
};

} // namespace rstr2
