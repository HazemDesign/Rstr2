// Rstr2 native core - Phase 3 entry point.
//
// Modes:
//   --ci                 CI smoke mode: print version and exit 0, no GPU.
//   (default)            Init D3D12/DXR, then loop: pull any scene the Blender
//                        addon published over shared memory, ray-trace it with
//                        the supplied camera, publish RGBA32F pixels to the
//                        frame shared memory, and idle until Ctrl+C so the
//                        addon can attach and read frames.

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
        "Rstr2Core - Phase 3 DXR renderer\n"
        "Usage: Rstr2Core [--width W] [--height H] [--shm NAME] [--ci]\n"
        "  --width  W   frame width  (default 960)\n"
        "  --height H   frame height (default 540)\n"
        "  --shm   NAME shared-memory mapping name for published frames (default Local\\Rstr2Frame_v1)\n"
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
        std::printf("Rstr2Core 0.3.0 (Phase 3) - CI mode, no GPU init\n");
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

    // Scene bridge (addon -> core). Created lazily by the addon; we open it
    // once it appears and poll it for updates each frame.
    rstr2::SceneMem scene(L"Local\\Rstr2Scene_v1");

    std::vector<float> pixels(width * height * 4);

    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    std::printf("Rstr2Core: rendering %dx%d, waiting for scene/ctrl-c\n", width, height);

    while (g_running.load()) {
        if (!scene.is_open()) scene.open();
        if (scene.is_open()) {
            rstr2::SceneData sd;
            if (scene.read_scene(sd)) {
                std::string se;
                if (!renderer.set_scene(sd, se)) {
                    std::fprintf(stderr, "Rstr2Core: set_scene failed: %s\n", se.c_str());
                }
            }
        }

        std::string re;
        if (!renderer.render_frame(pixels.data(), re)) {
            std::fprintf(stderr, "Rstr2Core: render failed: %s\n", re.c_str());
        } else {
            shm.publish_frame(pixels.data(), width, height);
        }

        Sleep(33); // ~30 fps; cheap for a single-bounce triangle soup.
    }

    std::printf("Rstr2Core: shutting down\n");
    return 0;
}
