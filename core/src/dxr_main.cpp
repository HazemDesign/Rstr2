// Rstr2 native core - DXR 1.1 entry point.
//
// Drop-in alternative to main.cpp (OptiX): same shared-memory loop, but traces
// with the DXR backend (rstr2::DxrRenderer) so it runs on any DXR-capable GPU.
// See main.cpp for the worker-thread / large-stack rationale.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>

#include <windows.h>
#include <cstdarg>
#include <process.h>

#include "shared_mem.h"
#include "dxr_renderer.h"

namespace {

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

std::atomic<bool> g_running{true};

BOOL WINAPI console_ctrl_handler(DWORD /*ctrl*/) {
    g_running.store(false);
    return TRUE;
}

void print_help() {
    std::printf(
        "Rstr2Dxr - Phase 3 DXR 1.1 renderer\n"
        "Usage: Rstr2Dxr [--width W] [--height H] [--shm NAME] [--ci]\n"
        "  --width  W   frame width  (default 960)\n"
        "  --height H   frame height (default 540)\n"
        "  --shm   NAME shared-memory mapping name for published frames (default Local\\Rstr2Frame_v1)\n"
        "  --ci         print version and exit 0 (no GPU)\n");
}

struct WorkerArgs {
    int width = 960;
    int height = 540;
    std::wstring shm_name = L"Local\\Rstr2Frame_v1";
    int exit_code = 0;
};

unsigned __stdcall worker_main(void* param) {
    WorkerArgs* a = static_cast<WorkerArgs*>(param);
    rlogf("Rstr2Dxr: worker thread started (3GB stack)\n");

    const uint32_t max_w = (a->width > 2048) ? (uint32_t)a->width : 2048u;
    const uint32_t max_h = (a->height > 1280) ? (uint32_t)a->height : 1280u;
    const size_t pixel_bytes = static_cast<size_t>(max_w) * max_h * 16u;
    const size_t shm_size = 256 + pixel_bytes;
    rlogf("Rstr2Dxr: frame shm max %ux%u (%zu KB)\n",
                 max_w, max_h, shm_size / 1024u);

    rstr2::SharedMem shm(a->shm_name, shm_size);
    if (!shm.valid()) {
        rlogf("Rstr2Dxr: failed to create shared memory '%ls' (%zu bytes)\n",
                     a->shm_name.c_str(), shm_size);
        a->exit_code = 1;
        return 1;
    }

    rstr2::DxrRenderer renderer;
    std::string err;
    if (!renderer.init(a->width, a->height, err)) {
        rlogf("Rstr2Dxr: renderer init failed: %s\n", err.c_str());
        a->exit_code = 1;
        return 1;
    }

    rstr2::SceneMem scene(L"Local\\Rstr2Scene_v1");

    std::vector<float> pixels(static_cast<size_t>(max_w) * max_h * 4u);
    int cur_w = a->width, cur_h = a->height;

    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    rlogf("Rstr2Dxr: rendering %dx%d, waiting for scene/ctrl-c\n", a->width, a->height);

    while (g_running.load()) {
        if (!scene.is_open()) scene.open();
        if (scene.is_open()) {
            rstr2::SceneData sd;
            if (scene.read_scene(sd)) {
                rlogf("Rstr2Dxr: scene received v=%u i=%u\n",
                             (unsigned)sd.vertices.size() / 3u, (unsigned)sd.indices.size());
                if (sd.render_width && sd.render_height) {
                    int rw = (sd.render_width > max_w) ? (int)max_w : (int)sd.render_width;
                    int rh = (sd.render_height > max_h) ? (int)max_h : (int)sd.render_height;
                    rw = (rw < 16) ? 16 : rw;
                    rh = (rh < 16) ? 16 : rh;
                    if (rw != cur_w || rh != cur_h) {
                        if (!renderer.resize(rw, rh, err)) {
                            rlogf("Rstr2Dxr: resize failed: %s\n", err.c_str());
                        } else {
                            cur_w = rw;
                            cur_h = rh;
                        }
                    }
                }
                std::string se;
                if (!renderer.set_scene(sd, se)) {
                    rlogf("Rstr2Dxr: set_scene failed: %s\n", se.c_str());
                }
            }
        }

        std::string re;
        if (!renderer.render_frame(pixels.data(), re)) {
            rlogf("Rstr2Dxr: render failed: %s\n", re.c_str());
        } else {
            shm.publish_frame(pixels.data(), cur_w, cur_h);
        }

        Sleep(33);
    }

    rlogf("Rstr2Dxr: shutting down\n");
    a->exit_code = 0;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    rlogf("Rstr2Dxr: process starting\n");

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
            rlogf("Rstr2Dxr: unknown argument '%s'\n", a.c_str());
            print_help();
            return 2;
        }
    }

    if (ci) {
        std::printf("Rstr2Dxr 0.3.0 (Phase 3, DXR 1.1) - CI mode, no GPU init\n");
        return 0;
    }

    if (width <= 0 || height <= 0) {
        rlogf("Rstr2Dxr: invalid dimensions %dx%d\n", width, height);
        return 2;
    }

    WorkerArgs args;
    args.width = width;
    args.height = height;
    args.shm_name = shm_name;

    const SIZE_T kThreadStack = 3u * 1024u * 1024u * 1024u; // 3 GB
    HANDLE hThread = reinterpret_cast<HANDLE>(
        _beginthreadex(nullptr, static_cast<unsigned>(kThreadStack),
                       worker_main, &args, 0, nullptr));
    if (hThread == nullptr) {
        rlogf("Rstr2Dxr: failed to create worker thread (%lu)\n", GetLastError());
        return 1;
    }

    WaitForSingleObject(hThread, INFINITE);
    CloseHandle(hThread);
    return args.exit_code;
}
