// Rstr2 native core - Phase 3 entry point.
//
// Modes:
//   --ci                 CI smoke mode: print version and exit 0, no GPU.
//   (default)            Init D3D12/DXR, then loop: pull any scene the Blender
//                        addon published over shared memory, ray-trace it with
//                        the supplied camera, publish RGBA32F pixels to the
//                        frame shared memory, and idle until Ctrl+C so the
//                        addon can attach and read frames.
//
// All DXR init and the render loop run on a worker thread with an EXPLICIT
// large stack. Some GPU drivers (notably new Blackwell DXR paths) consume a
// very large amount of stack per D3D12 call; a thread with a 512 MB stack is
// immune to the PE-header /STACK setting not taking effect and avoids the
// STATUS_STACK_OVERFLOW we otherwise hit during resource/descriptor creation.

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
#include "renderer.h"

namespace {

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

std::atomic<bool> g_running{true};

BOOL WINAPI console_ctrl_handler(DWORD /*ctrl*/) {
    g_running.store(false);
    return TRUE;
}

void print_help() {
    std::printf(
        "Rstr2Core - Phase 3 OptiX renderer\n"
        "Usage: Rstr2Core [--width W] [--height H] [--shm NAME] [--ci]\n"
        "  --width  W   frame width  (default 960)\n"
        "  --height H   frame height (default 540)\n"
        "  --shm   NAME shared-memory mapping name for published frames (default Local\\Rstr2Frame_v1)\n"
        "  --ci         print version and exit 0 (no GPU)\n");
}

// Arguments + result handed to the worker thread.
struct WorkerArgs {
    int width = 960;
    int height = 540;
    std::wstring shm_name = L"Local\\Rstr2Frame_v1";
    int exit_code = 0;
};

unsigned __stdcall worker_main(void* param) {
    WorkerArgs* a = static_cast<WorkerArgs*>(param);
    rlogf("Rstr2Core: worker thread started (3GB stack)\n");

    const size_t pixel_bytes = static_cast<size_t>(a->width) * static_cast<size_t>(a->height) * 16u;
    const size_t shm_size = 256 + pixel_bytes;

    rstr2::SharedMem shm(a->shm_name, shm_size);
    if (!shm.valid()) {
        rlogf("Rstr2Core: failed to create shared memory '%ls' (%zu bytes)\n",
                     a->shm_name.c_str(), shm_size);
        a->exit_code = 1;
        return 1;
    }

    rstr2::Renderer renderer;
    std::string err;
    if (!renderer.init(a->width, a->height, err)) {
        rlogf("Rstr2Core: renderer init failed: %s\n", err.c_str());
        a->exit_code = 1;
        return 1;
    }

    // Scene bridge (addon -> core). Created lazily by the addon; we open it
    // once it appears and poll it for updates each frame.
    rstr2::SceneMem scene(L"Local\\Rstr2Scene_v1");

    std::vector<float> pixels(static_cast<size_t>(a->width) * static_cast<size_t>(a->height) * 4u);

    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    rlogf("Rstr2Core: rendering %dx%d, waiting for scene/ctrl-c\n", a->width, a->height);

    while (g_running.load()) {
        if (!scene.is_open()) scene.open();
        if (scene.is_open()) {
            rstr2::SceneData sd;
            if (scene.read_scene(sd)) {
                rlogf("Rstr2Core: scene received v=%u i=%u\n",
                             (unsigned)sd.vertices.size() / 3u, (unsigned)sd.indices.size());
                std::string se;
                if (!renderer.set_scene(sd, se)) {
                    rlogf("Rstr2Core: set_scene failed: %s\n", se.c_str());
                }
            }
        }

        std::string re;
        if (!renderer.render_frame(pixels.data(), re)) {
            rlogf("Rstr2Core: render failed: %s\n", re.c_str());
        } else {
            shm.publish_frame(pixels.data(), a->width, a->height);
        }

        Sleep(33); // ~30 fps; cheap for a single-bounce triangle soup.
    }

    rlogf("Rstr2Core: shutting down\n");
    a->exit_code = 0;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // Diagnostics: the addon launches us windowless, so route everything to
    // stderr and keep it unbuffered. core_proc.py redirects stderr to
    // bin/Rstr2Core.log so the user can read why we failed to start.
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    rlogf("Rstr2Core: process starting\n");

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
            rlogf("Rstr2Core: unknown argument '%s'\n", a.c_str());
            print_help();
            return 2;
        }
    }

    if (ci) {
        std::printf("Rstr2Core 0.3.0 (Phase 3, OptiX) - CI mode, no GPU init\n");
        return 0;
    }

    if (width <= 0 || height <= 0) {
        rlogf("Rstr2Core: invalid dimensions %dx%d\n", width, height);
        return 2;
    }

    WorkerArgs args;
    args.width = width;
    args.height = height;
    args.shm_name = shm_name;

    // Run all DXR init + render loop on a worker thread with an explicit large
    // stack (3 GB) so per-call driver stack usage cannot overflow us. The RTX
    // 5050 (Blackwell) D3D12 driver path consumes an abnormal amount of stack
    // per CreateCommittedResource/CreateUnorderedAccessView call; 3 GB gives us
    // ample headroom for the whole init sequence.
    const SIZE_T kThreadStack = 3u * 1024u * 1024u * 1024u; // 3 GB
    HANDLE hThread = reinterpret_cast<HANDLE>(
        _beginthreadex(nullptr, static_cast<unsigned>(kThreadStack),
                      worker_main, &args, 0, nullptr));
    if (hThread == nullptr) {
        rlogf("Rstr2Core: failed to create worker thread (%lu)\n", GetLastError());
        return 1;
    }

    WaitForSingleObject(hThread, INFINITE);
    CloseHandle(hThread);
    return args.exit_code;
}
