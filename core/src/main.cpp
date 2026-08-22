// Rstr2 native core - Phase 2 entry point.
//
// Modes:
//   --ci                 CI smoke mode: print version and exit 0, no GPU.
//   (default)            Init D3D12/DXR, ray-trace one hardcoded triangle,
//                        publish RGBA32F pixels to Win32 shared memory, then
//                        idle until Ctrl+C so the Blender addon can attach.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>

#include <windows.h>

#include "shared_mem.h"
#include "renderer.h"

namespace {

std::atomic<bool> g_running{true};

BOOL WINAPI console_ctrl_handler(DWORD /*ctrl*/) {
    g_running.store(false);
    return TRUE;
}

void print_help() {
    std::printf(
        "Rstr2Core - Phase 2 DXR renderer\n"
        "Usage: Rstr2Core [--width W] [--height H] [--shm NAME] [--ci]\n"
        "  --width  W   frame width  (default 960)\n"
        "  --height H   frame height (default 540)\n"
        "  --shm   NAME shared-memory mapping name (default Local\\Rstr2Frame_v1)\n"
        "  --ci         print version and exit 0 (no GPU)\n");
}

} // namespace

int main(int argc, char** argv) {
    int width = 960;
    int height = 540;
    std::wstring shm_name = L"Local\\Rstr2Frame_v1";
    bool ci = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ci") {
            ci = true;
        } else if (a == "--width" && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        } else if (a == "--height" && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        } else if (a == "--shm" && i + 1 < argc) {
            std::string n = argv[++i];
            shm_name = std::wstring(n.begin(), n.end());
        } else if (a == "--help" || a == "-h") {
            print_help();
            return 0;
        } else {
            std::fprintf(stderr, "Rstr2Core: unknown argument '%s'\n", a.c_str());
            print_help();
            return 2;
        }
    }

    if (ci) {
        std::printf("Rstr2Core 0.2.0 (Phase 2) - CI mode, no GPU init\n");
        return 0;
    }

    if (width <= 0 || height <= 0) {
        std::fprintf(stderr, "Rstr2Core: invalid dimensions %dx%d\n", width, height);
        return 2;
    }

    const size_t pixel_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 16u;
    const size_t shm_size = 256 + pixel_bytes;

    rstr2::SharedMem shm(shm_name, shm_size);
    if (!shm.valid()) {
        std::fprintf(stderr, "Rstr2Core: failed to create shared memory '%ls' (%zu bytes)\n",
                     shm_name.c_str(), shm_size);
        return 1;
    }

    rstr2::Renderer renderer;
    std::string err;
    if (!renderer.init(width, height, err)) {
        std::fprintf(stderr, "Rstr2Core: renderer init failed: %s\n", err.c_str());
        return 1;
    }

    std::vector<float> pixels(width * height * 4);
    if (!renderer.render_frame(pixels.data(), err)) {
        std::fprintf(stderr, "Rstr2Core: render failed: %s\n", err.c_str());
        return 1;
    }

    shm.publish_frame(pixels.data(), width, height);
    std::printf("Rstr2Core: frame published %dx%d\n", width, height);

    // Idle until Ctrl+C so the Blender addon can attach and read frames.
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    while (g_running.load()) {
        Sleep(1000);
    }

    std::printf("Rstr2Core: shutting down\n");
    return 0;
}
