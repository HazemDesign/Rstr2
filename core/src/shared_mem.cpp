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

// ----------------------------------------------------------------------
// SceneMem (reader side)
// ----------------------------------------------------------------------
SceneMem::SceneMem(const std::wstring& name) : name_(name) {}

SceneMem::~SceneMem() { close(); }

bool SceneMem::open() {
    close();
    HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, 0, name_.empty() ? nullptr : name_.c_str());
    if (mapping == nullptr || mapping == INVALID_HANDLE_VALUE) return false;
    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(mapping);
        return false;
    }
    mapping_ = mapping;
    view_ = view;
    return true;
}

void SceneMem::close() {
    if (view_ != nullptr) { UnmapViewOfFile(view_); view_ = nullptr; }
    if (mapping_ != nullptr) { CloseHandle(mapping_); mapping_ = nullptr; }
}

bool SceneMem::read_scene(SceneData& out) {
    if (view_ == nullptr) return false;

    const SceneHeader* hdr = reinterpret_cast<const SceneHeader*>(view_);

    // Read the epoch/flags with acquire so the data writes are visible.
    uint32_t epoch = std::atomic_ref<const uint32_t>(hdr->epoch).load(std::memory_order_acquire);
    uint32_t ready = std::atomic_ref<const uint32_t>(hdr->ready).load(std::memory_order_acquire);
    uint32_t writing = std::atomic_ref<const uint32_t>(hdr->writing).load(std::memory_order_acquire);
    if (ready != 1 || writing != 0 || epoch == last_epoch_) return false;

    if (!consistent_copy(out)) return false;

    // Re-check epoch + writing after the copy to detect a torn update.
    uint32_t epoch2 = std::atomic_ref<const uint32_t>(hdr->epoch).load(std::memory_order_acquire);
    uint32_t writing2 = std::atomic_ref<const uint32_t>(hdr->writing).load(std::memory_order_acquire);
    if (writing2 != 0 || epoch2 != epoch) return false;

    last_epoch_ = epoch;
    return true;
}

bool SceneMem::consistent_copy(SceneData& out) {
    const SceneHeader* hdr = reinterpret_cast<const SceneHeader*>(view_);
    if (hdr->magic != kSceneMagic || hdr->version != kSceneVersion) return false;

    const uint32_t vcount = hdr->vertex_count;
    const uint32_t icount = hdr->index_count;
    if (vcount == 0 || icount == 0 || (icount % 3) != 0) return false;

    const size_t vbytes = static_cast<size_t>(vcount) * 3u * sizeof(float);
    const size_t ibytes = static_cast<size_t>(icount) * sizeof(uint32_t);
    const uint32_t lcount = hdr->light_count;
    const size_t lbytes = static_cast<size_t>(lcount) * 16u * sizeof(float);
    const size_t abytes = static_cast<size_t>(vcount) * 3u * sizeof(float);
    if (vbytes + ibytes + lbytes + abytes > kMaxSceneBytes) return false;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(view_);
    const float* vptr = reinterpret_cast<const float*>(base + kSceneHeaderSize);
    const uint32_t* iptr = reinterpret_cast<const uint32_t*>(
        base + kSceneHeaderSize + vbytes);
    const Light* lptr = reinterpret_cast<const Light*>(
        base + kSceneHeaderSize + vbytes + ibytes);
    const float* aptr = reinterpret_cast<const float*>(
        base + kSceneHeaderSize + vbytes + ibytes + lbytes);

    out.vertices.assign(vptr, vptr + static_cast<size_t>(vcount) * 3u);
    out.indices.assign(iptr, iptr + icount);
    out.lights.assign(lptr, lptr + lcount);
    out.albedos.assign(aptr, aptr + static_cast<size_t>(vcount) * 3u);

    out.flags = hdr->flags;
    out.exposure = hdr->exposure;
    out.taa_history = hdr->taa_history;

    for (int i = 0; i < 3; ++i) {
        out.cam_origin[i] = hdr->cam_origin[i];
        out.cam_right[i] = hdr->cam_right[i];
        out.cam_up[i] = hdr->cam_up[i];
        out.cam_forward[i] = hdr->cam_forward[i];
    }
    out.cam_tan_half_fov_y = hdr->cam_tan_half_fov_y;
    return true;
}

} // namespace rstr2
