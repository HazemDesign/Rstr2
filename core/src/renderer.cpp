// Rstr2 Phase 3 - OptiX renderer implementation. See renderer.h.
//
// Single-bounce triangle-soup path tracer built on OptiX 7+/9. Geometry is
// uploaded as a single GAS (bottom-level acceleration structure) and traced
// directly; the camera basis + buffers are passed through the launch params.

#include "renderer.h"
#include "optix_params.h"

// OptiX 9 loads optix.64.dll from the NVIDIA driver at runtime via the stub
// table (no import lib, nothing to ship). The function-table instance must be
// defined in exactly one TU by including optix_function_table_definition.h,
// which provides the symbol that optixInit() populates.
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda.h>

#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace rstr2 {

namespace {

// Timestamped diagnostic line to stderr (the addon redirects stderr to
// bin/Rstr2Core.log so failures are readable).
static void rlogf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "Rstr2Core: %s\n", buf);
    fflush(stderr);
}

static void optix_log_cb(unsigned int level, const char* tag,
                         const char* message, void* /*data*/) {
    (void)level; (void)tag;
    fprintf(stderr, "Rstr2Core: [optix] %s\n", message);
    fflush(stderr);
}

#define CU_CHECK(call)                                                        \
    do {                                                                      \
        CUresult _r = (call);                                                 \
        if (_r != CUDA_SUCCESS) {                                             \
            const char* _name = nullptr;                                       \
            cuGetErrorName(_r, &_name);                                        \
            rlogf("CUDA error %s (%d) at %s:%d in %s",                        \
                  _name ? _name : "?", (int)_r, __FILE__, __LINE__, #call);   \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define OPTIX_CHECK(call)                                                     \
    do {                                                                      \
        OptixResult _r = (call);                                              \
        if (_r != OPTIX_SUCCESS) {                                            \
            rlogf("OptiX error %s (%d) at %s:%d in %s",                       \
                  optixGetErrorName(_r), (int)_r, __FILE__, __LINE__, #call); \
            return false;                                                     \
        }                                                                     \
    } while (0)

// Default fallback scene: a single triangle + the matching camera. Matches the
// previous DXR default so we always have something to render before/without
// the addon scene bridge.
SceneData make_default_scene() {
    SceneData s;
    s.vertices = {
        -0.7f, -0.5f, 0.0f,
         0.7f, -0.5f, 0.0f,
         0.0f,  0.7f, 0.0f,
    };
    s.indices = {0, 1, 2};
    s.cam_origin[0] = 0.0f; s.cam_origin[1] = 0.6f; s.cam_origin[2] = -2.2f;
    s.cam_right[0] = 1.0f; s.cam_right[1] = 0.0f; s.cam_right[2] = 0.0f;
    s.cam_up[0] = 0.0f; s.cam_up[1] = 1.0f; s.cam_up[2] = 0.0f;
    s.cam_forward[0] = 0.0f; s.cam_forward[1] = 0.0f; s.cam_forward[2] = 1.0f;
    s.cam_tan_half_fov_y = std::tan(0.45f);
    return s;
}

} // namespace

struct Renderer::Impl {
    CUcontext        cuda_ctx       = nullptr;
    CUstream         stream         = nullptr;
    OptixDeviceContext optix_ctx    = nullptr;

    OptixModule      module         = nullptr;
    OptixProgramGroup raygen_pg     = nullptr;
    OptixProgramGroup miss_pg       = nullptr;
    OptixProgramGroup hitgroup_pg   = nullptr;
    OptixPipeline    pipeline       = nullptr;
    OptixShaderBindingTable sbt     = {};

    CUdeviceptr      d_sbt_records  = 0;   // raygen + miss + hit records
    CUdeviceptr      d_output       = 0;   // RGBA32F, width*height*4
    CUdeviceptr      d_vertices     = 0;
    CUdeviceptr      d_indices      = 0;
    size_t           vert_bytes     = 0;
    size_t           idx_bytes      = 0;
    CUdeviceptr      d_gas          = 0;   // acceleration structure
    CUdeviceptr      d_gas_scratch  = 0;
    size_t           gas_bytes      = 0;
    size_t           gas_scratch_bytes = 0;
    uint64_t         traversable    = 0;   // OptixTraversableHandle

    CUdeviceptr      d_params       = 0;   // launch params buffer

    // RTXDI (ReSTIR DI) state.
    CUdeviceptr      d_gbuf         = 0;   // 3*N float4: [pos,hit],[normal],[albedo]
    CUdeviceptr      d_res[2]        = {0, 0}; // two reservoir buffers (ping-pong)
    CUdeviceptr      d_lights       = 0;   // typed-light pool
    CUdeviceptr      d_albedos      = 0;   // per-vertex rgb (may be null)
    CUdeviceptr      d_accum        = 0;   // HDR TAA accumulation buffer
    size_t           gbuf_bytes     = 0;
    size_t           res_bytes      = 0;
    size_t           light_bytes    = 0;
    size_t           albedo_bytes   = 0;
    uint32_t         frame_index    = 0;
    bool             scene_dirty    = true;

    ~Impl() {
        if (d_output)       cuMemFree(d_output);
        if (d_vertices)     cuMemFree(d_vertices);
        if (d_indices)      cuMemFree(d_indices);
        if (d_gas)          cuMemFree(d_gas);
        if (d_gas_scratch)  cuMemFree(d_gas_scratch);
        if (d_sbt_records)  cuMemFree(d_sbt_records);
        if (d_params)       cuMemFree(d_params);
        if (d_gbuf)         cuMemFree(d_gbuf);
        if (d_res[0])       cuMemFree(d_res[0]);
        if (d_res[1])       cuMemFree(d_res[1]);
        if (d_lights)       cuMemFree(d_lights);
        if (d_albedos)      cuMemFree(d_albedos);
        if (d_accum)        cuMemFree(d_accum);
        if (pipeline)       optixPipelineDestroy(pipeline);
        if (hitgroup_pg)    optixProgramGroupDestroy(hitgroup_pg);
        if (miss_pg)        optixProgramGroupDestroy(miss_pg);
        if (raygen_pg)      optixProgramGroupDestroy(raygen_pg);
        if (module)         optixModuleDestroy(module);
        if (optix_ctx)      optixDeviceContextDestroy(optix_ctx);
        if (stream)         cuStreamDestroy(stream);
        if (cuda_ctx)       cuCtxDestroy(cuda_ctx);
    }

    bool init_cuda(std::string& error) {
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) {
            const char* n = nullptr; cuGetErrorName(r, &n);
            error = std::string("Rstr2: cuInit failed (") + (n ? n : "?") +
                    "). Is a CUDA-capable GPU + driver present?";
            return false;
        }
        // Load the OptiX runtime from the NVIDIA driver (optix.64.dll).
        OptixResult oi = optixInit();
        if (oi != OPTIX_SUCCESS) {
            error = "Rstr2: optixInit failed. Is an NVIDIA R570+ driver with "
                    "OptiX support installed?";
            return false;
        }
        CUdevice device = 0;
        CU_CHECK(cuDeviceGet(&device, 0));
        CU_CHECK(cuCtxCreate(&cuda_ctx, 0, device));
        CU_CHECK(cuStreamCreate(&stream, 0));

        OptixDeviceContextOptions opts = {};
        opts.logCallbackFunction = optix_log_cb;
        opts.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuda_ctx, &opts, &optix_ctx));
        return true;
    }

    bool create_module_and_pipeline(std::string& error) {
        // Locate the compiled kernel PTX next to the executable.
        wchar_t mod[1024] = {};
        DWORD n = GetModuleFileNameW(nullptr, mod, (DWORD)_countof(mod));
        if (n == 0 || n >= _countof(mod)) {
            error = "Rstr2: failed to resolve executable path.";
            return false;
        }
        std::wstring path(mod);
        auto slash = path.find_last_of(L"\\/");
        if (slash != std::wstring::npos) path = path.substr(0, slash + 1);
        path += L"optix_kernels.ptx";

        std::ifstream f(path, std::ios::binary);
        if (!f) {
            error = "Rstr2: optix_kernels.ptx not found next to the executable. "
                    "Build the project so shaders/optix_kernels.cu is compiled to "
                    "optix_kernels.ptx.";
            return false;
        }
        f.seekg(0, std::ios::end);
        std::streamoff sz = f.tellg();
        if (sz <= 0) { error = "Rstr2: optix_kernels.ptx is empty."; return false; }
        f.seekg(0, std::ios::beg);
        std::string ptx(static_cast<size_t>(sz), '\0');
        f.read(&ptx[0], sz);
        if (!f) { error = "Rstr2: failed to read optix_kernels.ptx."; return false; }

        OptixModuleCompileOptions mco = {};
        mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        OptixPipelineCompileOptions pco = {};
        pco.usesMotionBlur = 0;
        pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pco.numPayloadValues = 4;
        pco.numAttributeValues = 2;
        pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pco.pipelineLaunchParamsVariableName = "params";

        char log[8192] = {};
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixModuleCreate(optix_ctx, &mco, &pco,
                                     ptx.c_str(), ptx.size(), log,
                                     &log_size, &module));
        if (log[0]) rlogf("Rstr2Core: [optix module] %s", log);

        OptixProgramGroupDesc descs[4] = {};
        descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descs[0].raygen.module = module;
        descs[0].raygen.entryFunctionName = "__raygen__rg_primary";

        descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descs[1].raygen.module = module;
        descs[1].raygen.entryFunctionName = "__raygen__rg_shade";

        descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descs[2].miss.module = module;
        descs[2].miss.entryFunctionName = "__miss__ms";

        descs[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descs[3].hitgroup.moduleCH = module;
        descs[3].hitgroup.entryFunctionNameCH = "__closesthit__ch";

        OptixProgramGroup pgs[4] = {};
        OptixProgramGroupOptions pg_opts = {};
        OPTIX_CHECK(optixProgramGroupCreate(optix_ctx, descs, 4, &pg_opts,
                                            log, &log_size, pgs));
        if (log[0]) rlogf("Rstr2Core: [optix pgs] %s", log);
        raygen_pg = pgs[0];
        miss_pg = pgs[2];
        hitgroup_pg = pgs[3];

        OptixPipelineLinkOptions link_opts = {};
        link_opts.maxTraceDepth = 1;
        OPTIX_CHECK(optixPipelineCreate(optix_ctx, &pco, &link_opts, pgs, 4,
                                         log, &log_size, &pipeline));
        if (log[0]) rlogf("Rstr2Core: [optix pipeline] %s", log);

        // Shader binding table: [rg_primary, rg_shade, miss, hit].
        const size_t rec = OPTIX_SBT_RECORD_HEADER_SIZE;
        char rg0[OPTIX_SBT_RECORD_HEADER_SIZE];
        char rg1[OPTIX_SBT_RECORD_HEADER_SIZE];
        char ms[OPTIX_SBT_RECORD_HEADER_SIZE];
        char hg[OPTIX_SBT_RECORD_HEADER_SIZE];
        OPTIX_CHECK(optixSbtRecordPackHeader(pgs[0], rg0));
        OPTIX_CHECK(optixSbtRecordPackHeader(pgs[1], rg1));
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, ms));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, hg));

        CU_CHECK(cuMemAlloc(&d_sbt_records, rec * 4));
        CU_CHECK(cuMemcpyHtoD(d_sbt_records + rec * 0, rg0, rec));
        CU_CHECK(cuMemcpyHtoD(d_sbt_records + rec * 1, rg1, rec));
        CU_CHECK(cuMemcpyHtoD(d_sbt_records + rec * 2, ms, rec));
        CU_CHECK(cuMemcpyHtoD(d_sbt_records + rec * 3, hg, rec));

        sbt.raygenRecord = d_sbt_records;                       // toggled per launch
        sbt.missRecordBase = d_sbt_records + rec * 2;
        sbt.missRecordStrideInBytes = (unsigned int)rec;
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_sbt_records + rec * 3;
        sbt.hitgroupRecordStrideInBytes = (unsigned int)rec;
        sbt.hitgroupRecordCount = 1;

        rlogf("Rstr2Core: module + pipeline + sbt ok\n");
        return true;
    }
};

Renderer::Renderer() = default;

Renderer::~Renderer() {
    delete impl_;
}

bool Renderer::init(int width, int height, std::string& error) {
    width_ = width;
    height_ = height;
    rlogf("Rstr2Core: OptiX init begin (%dx%d)\n", width, height);

    impl_ = new Impl();
    if (!impl_->init_cuda(error)) return false;
    rlogf("Rstr2Core: cuda/optix ctx ok\n");

    // Allocate the RGBA32F output + launch params buffers.
    const size_t out_bytes = static_cast<size_t>(width) * height * 16u;
    CU_CHECK(cuMemAlloc(&impl_->d_output, out_bytes));
    CU_CHECK(cuMemAlloc(&impl_->d_params, sizeof(Params)));

    // G-buffer (3 float4/pixel) + reservoir ping-pong + TAA accumulation.
    const size_t gbuf_bytes = static_cast<size_t>(width) * height * 3u * 16u;
    const size_t res_bytes = static_cast<size_t>(width) * height * sizeof(Reservoir);
    const size_t accum_bytes = static_cast<size_t>(width) * height * 16u;
    CU_CHECK(cuMemAlloc(&impl_->d_gbuf, gbuf_bytes));
    CU_CHECK(cuMemAlloc(&impl_->d_res[0], res_bytes));
    CU_CHECK(cuMemAlloc(&impl_->d_res[1], res_bytes));
    CU_CHECK(cuMemAlloc(&impl_->d_accum, accum_bytes));
    // Reservoirs must start empty (M=0) so the first frame has no bogus
    // temporal neighbour; accumulation starts black and fades in.
    CU_CHECK(cuMemsetD8(impl_->d_res[0], 0, res_bytes));
    CU_CHECK(cuMemsetD8(impl_->d_res[1], 0, res_bytes));
    CU_CHECK(cuMemsetD8(impl_->d_accum, 0, accum_bytes));
    impl_->gbuf_bytes = gbuf_bytes;
    impl_->res_bytes = res_bytes;

    // Module + program groups + pipeline.
    if (!impl_->create_module_and_pipeline(error)) return false;

    // Default fallback scene so we always have geometry to trace.
    scene_ = make_default_scene();
    have_scene_ = true;
    if (!set_scene(scene_, error)) return false;

    rlogf("Rstr2Core: OptiX init complete\n");
    return true;
}

bool Renderer::set_scene(const SceneData& scene, std::string& error) {
    scene_ = scene;
    have_scene_ = true;

    Impl* im = impl_;
    if (!im) { error = "Rstr2: renderer not initialized."; return false; }

    const size_t vcount = scene_.vertices.size() / 3;
    const size_t icount = scene_.indices.size();
    if (vcount == 0 || icount == 0 || (icount % 3) != 0) {
        error = "Rstr2: scene has no triangles.";
        return false;
    }
    rlogf("Rstr2Core: set_scene v=%zu i=%zu l=%zu alb=%zu flg=%u exp=%.2f hist=%.1f",
          vcount, icount, scene_.lights.size(), scene_.albedos.size() / 3u,
          scene_.flags, scene_.exposure, scene_.taa_history);

    // Free previous buffers.
    if (im->d_vertices) { cuMemFree(im->d_vertices); im->d_vertices = 0; }
    if (im->d_indices)  { cuMemFree(im->d_indices);  im->d_indices = 0; }
    if (im->d_gas)      { cuMemFree(im->d_gas);      im->d_gas = 0; }
    if (im->d_gas_scratch) { cuMemFree(im->d_gas_scratch); im->d_gas_scratch = 0; }

    im->vert_bytes = vcount * sizeof(Vec3F);
    im->idx_bytes  = icount * sizeof(unsigned int);

    CU_CHECK(cuMemAlloc(&im->d_vertices, im->vert_bytes));
    CU_CHECK(cuMemAlloc(&im->d_indices, im->idx_bytes));
    CU_CHECK(cuMemcpyHtoD(im->d_vertices, scene_.vertices.data(), im->vert_bytes));
    CU_CHECK(cuMemcpyHtoD(im->d_indices, scene_.indices.data(), im->idx_bytes));

    // Lights (RTXDI candidate pool). Reallocated if the count changed.
    const size_t lbytes = scene_.lights.size() * sizeof(Light);
    if (im->d_lights) { cuMemFree(im->d_lights); im->d_lights = 0; im->light_bytes = 0; }
    if (lbytes > 0) {
        CU_CHECK(cuMemAlloc(&im->d_lights, lbytes));
        CU_CHECK(cuMemcpyHtoD(im->d_lights, scene_.lights.data(), lbytes));
        im->light_bytes = lbytes;
    }

    // Per-vertex albedo (may be absent -> kernel falls back to a default).
    if (im->d_albedos) { cuMemFree(im->d_albedos); im->d_albedos = 0; im->albedo_bytes = 0; }
    const size_t abytes = scene_.albedos.size() * sizeof(float);
    if (abytes > 0 && abytes == im->vert_bytes) {
        CU_CHECK(cuMemAlloc(&im->d_albedos, abytes));
        CU_CHECK(cuMemcpyHtoD(im->d_albedos, scene_.albedos.data(), abytes));
        im->albedo_bytes = abytes;
    }

    // Any new epoch invalidates temporal history (geometry/camera/lights/
    // settings changed) -> the next frame resets TAA accumulation.
    im->scene_dirty = true;

    // Build a single triangle-array GAS from the indexed mesh. Disable face
    // culling so geometry is visible regardless of index winding.
    unsigned int tri_flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &im->d_vertices;
    build_input.triangleArray.numVertices = (unsigned int)vcount;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(Vec3F);
    build_input.triangleArray.indexBuffer = im->d_indices;
    build_input.triangleArray.numIndexTriplets = (unsigned int)(icount / 3);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = 0;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = 0;
    build_input.triangleArray.flags = &tri_flags;

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(im->optix_ctx, &accel_opts,
                                             &build_input, 1, &gas_sizes));
    im->gas_bytes = gas_sizes.outputSizeInBytes;
    im->gas_scratch_bytes = gas_sizes.tempSizeInBytes;

    CU_CHECK(cuMemAlloc(&im->d_gas, im->gas_bytes));
    CU_CHECK(cuMemAlloc(&im->d_gas_scratch, im->gas_scratch_bytes));

    OptixTraversableHandle handle = 0;
    OPTIX_CHECK(optixAccelBuild(im->optix_ctx, im->stream, &accel_opts,
                                &build_input, 1,
                                im->d_gas_scratch, im->gas_scratch_bytes,
                                im->d_gas, im->gas_bytes,
                                &handle, nullptr, 0));
    im->traversable = handle;
    CU_CHECK(cuStreamSynchronize(im->stream));

    rlogf("Rstr2Core: GAS built (handle=%llu bytes=%zu scratch=%zu)\n",
          (unsigned long long)handle, im->gas_bytes, im->gas_scratch_bytes);
    return true;
}

bool Renderer::render_frame(float* out_pixels, std::string& error) {
    Impl* im = impl_;
    if (!im || !im->optix_ctx) { error = "Rstr2: renderer not initialized."; return false; }

    // Ping-pong the two reservoir buffers across frames.
    const int cur = static_cast<int>(im->frame_index & 1u);
    const int prev = cur ^ 1;
    const size_t rec = OPTIX_SBT_RECORD_HEADER_SIZE;

    Params p;
    memset(&p, 0, sizeof(p));
    p.image = reinterpret_cast<Vec4F*>(im->d_output);
    p.width = static_cast<unsigned int>(width_);
    p.height = static_cast<unsigned int>(height_);
    p.vertices = reinterpret_cast<Vec3F*>(im->d_vertices);
    p.indices = reinterpret_cast<unsigned int*>(im->d_indices);
    p.albedos = (im->albedo_bytes > 0)
        ? reinterpret_cast<Vec3F*>(im->d_albedos) : nullptr;
    p.cam_origin = { scene_.cam_origin[0], scene_.cam_origin[1], scene_.cam_origin[2] };
    p.cam_right  = { scene_.cam_right[0],  scene_.cam_right[1],  scene_.cam_right[2] };
    p.cam_up     = { scene_.cam_up[0],     scene_.cam_up[1],     scene_.cam_up[2] };
    p.cam_forward= { scene_.cam_forward[0],scene_.cam_forward[1],scene_.cam_forward[2] };
    p.cam_tan_half_fov_y = scene_.cam_tan_half_fov_y;
    p.handle = im->traversable;

    // RTXDI state.
    p.lights = reinterpret_cast<Light*>(im->d_lights);
    p.light_count = static_cast<unsigned int>(scene_.lights.size());
    p.gbuf = reinterpret_cast<Vec4F*>(im->d_gbuf);
    p.reservoirs = reinterpret_cast<Reservoir*>(im->d_res[cur]);
    p.prev_reservoirs = reinterpret_cast<Reservoir*>(im->d_res[prev]);
    p.frame_index = im->frame_index;

    // TAA / display state. Subpixel jitter from a low-discrepancy frame
    // sequence; history resets on the first frame after any scene change
    // (or when TAA is disabled via the addon settings).
    const bool taa_on = (scene_.flags & kSceneFlagTaa) != 0;
    auto frac = [](float x) { return x - std::floor(x); };
    if (taa_on && !im->scene_dirty) {
        p.jitter_x = frac(im->frame_index * 0.7548776662f) - 0.5f;
        p.jitter_y = frac(im->frame_index * 0.5698402909f) - 0.5f;
    }
    float hist = scene_.taa_history;
    if (hist < 1.0f) hist = 1.0f;
    p.accum_alpha = (taa_on && !im->scene_dirty) ? (1.0f / hist) : 1.0f;
    p.exposure = (scene_.exposure > 0.0f) ? scene_.exposure : 1.0f;
    p.taa_clamp = 10.0f;   // blRstr-parity default firefly clamp
    p.accum = reinterpret_cast<Vec4F*>(im->d_accum);

    // World / film state.
    p.world_r = scene_.world_color[0];
    p.world_g = scene_.world_color[1];
    p.world_b = scene_.world_color[2];
    p.world_strength = scene_.world_strength;
    p.film_transparent =
        (scene_.flags & kSceneFlagFilmTransparent) ? 1u : 0u;
    im->scene_dirty = false;

    CU_CHECK(cuMemcpyHtoD(im->d_params, &p, sizeof(Params)));

    // Pass 1: primary ray + ReSTIR DI reservoir build (temporal reuse).
    im->sbt.raygenRecord = im->d_sbt_records + rec * 0;
    OPTIX_CHECK(optixLaunch(im->pipeline, im->stream, im->d_params,
                            sizeof(Params), &im->sbt,
                            static_cast<unsigned int>(width_),
                            static_cast<unsigned int>(height_), 1));
    CU_CHECK(cuStreamSynchronize(im->stream));

    // Pass 2: shade using the selected light + a single shadow ray.
    im->sbt.raygenRecord = im->d_sbt_records + rec * 1;
    OPTIX_CHECK(optixLaunch(im->pipeline, im->stream, im->d_params,
                            sizeof(Params), &im->sbt,
                            static_cast<unsigned int>(width_),
                            static_cast<unsigned int>(height_), 1));
    CU_CHECK(cuStreamSynchronize(im->stream));

    const size_t out_bytes = static_cast<size_t>(width_) * height_ * 16u;
    CU_CHECK(cuMemcpyDtoH(out_pixels, im->d_output, out_bytes));

    // Temporary diagnostics: dump launch state + a probed pixel for the
    // first few frames after each scene change.
    if (im->frame_index < 3u || im->scene_dirty) {
        const size_t probe_idx =
            (static_cast<size_t>(height_ / 2) * width_ + width_ / 2) * 4u;
        float probe[4] = { -1, -1, -1, -1 };
        float acc_probe[4] = { -1, -1, -1, -1 };
        cuMemcpyDtoH(probe, im->d_output + probe_idx * sizeof(float), sizeof(probe));
        cuMemcpyDtoH(acc_probe, im->d_accum + probe_idx * sizeof(float), sizeof(acc_probe));
        rlogf("Rstr2Core: f=%u cnt=%u alb=%d jit=%.3f,%.3f a=%.3f exp=%.2f "
              "flg=%u hist=%.1f px=%.3f,%.3f,%.3f acc=%.3f,%.3f,%.3f",
              im->frame_index, p.light_count, im->albedo_bytes > 0 ? 1 : 0,
              p.jitter_x, p.jitter_y, p.accum_alpha, p.exposure,
              scene_.flags, scene_.taa_history,
              probe[0], probe[1], probe[2], acc_probe[0], acc_probe[1], acc_probe[2]);
    }

    im->frame_index++;
    return true;
}

} // namespace rstr2
