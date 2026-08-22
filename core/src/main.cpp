// Rstr2 native core - Phase 1 stub.
// CI builds this to prove the toolchain. D3D12/DXR arrives in Phase 2:
// device init, DXR accel structs, ReSTIR DI, shared-memory frame bridge.

#include <cstdio>

int main(int argc, char** argv) {
    std::printf("Rstr2Core stub v0.1.0 (Phase 1)\n");
    std::printf("args:");
    for (int i = 1; i < argc; ++i) std::printf(" %s", argv[i]);
    std::printf("\n");
    return 0;
}
