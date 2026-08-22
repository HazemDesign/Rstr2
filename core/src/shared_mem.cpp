// Rstr2 Phase 2 - shared memory frame bridge (writer side). See shared_mem.h.

#include "shared_mem.h"

#include <windows.h>
#include <cstring>
#include <atomic>

namespace rstr2 {

SharedMem::SharedMem(const std::wstring& name, size_t total_size) : size_(total_size) {
    HANDLE mapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE,   // pagefile-backed
        nullptr,                 // default security
        PAGE_READWRITE,
        0,                       // high 32 bits of size (size fits in 32 bits for our use)
        static_cast<DWORD>(total_size),
        name.empty() ? nullptr : name.c_str());

    if (mapping == nullptr) {
        view_ = nullptr;
        return;
    }

    void* view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
    if (view == nullptr) {
        CloseHandle(mapping);
        view_ = nullptr;
        return;
    }

    // Keep the mapping handle alive for the lifetime of the process.
    mapping_ = mapping;
    view_ = view;

    // Initialize header.
    FrameHeader* hdr = reinterpret_cast<FrameHeader*>(view_);
    hdr->magic = 0x32525352u;
    hdr->version = 1;
    hdr->width = 0;
    hdr->height = 0;
    hdr->frame_index = 0;
    hdr->state = 0;
    hdr->pixel_offset = 256;
    hdr->pixel_size = 0;
    std::memset(hdr->reserved, 0, sizeof(hdr->reserved));
}

SharedMem::~SharedMem() {
    if (view_ != nullptr) UnmapViewOfFile(view_);
    if (mapping_ != nullptr) CloseHandle(mapping_);
}

void SharedMem::publish_frame(const float* pixels, int width, int height) {
    if (view_ == nullptr) return;

    FrameHeader* hdr = reinterpret_cast<FrameHeader*>(view_);
    hdr->width = width;
    hdr->height = height;
    hdr->pixel_offset = 256;
    hdr->pixel_size = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 16u;

    const size_t pixel_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 16u;

    // 1. Copy pixels into the view at pixel_offset.
    std::memcpy(reinterpret_cast<uint8_t*>(view_) + hdr->pixel_offset, pixels, pixel_bytes);

    // 2. Release fence so the pixel writes are visible before we flip state.
    std::atomic_thread_fence(std::memory_order_release);

    // 3. Mark state ready.
    hdr->state = 1;

    // 4. Increment frame_index LAST, with release semantics, so a reader that
    //    observes a new frame_index also observes state=1 and the pixels.
    uint32_t next = ++frame_index_;
    std::atomic_ref<uint32_t>(hdr->frame_index).store(next, std::memory_order_release);
}

} // namespace rstr2
