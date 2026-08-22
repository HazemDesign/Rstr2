// Rstr2 Phase 2 - shared memory frame bridge (writer side).
//
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

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace rstr2 {

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

} // namespace rstr2
