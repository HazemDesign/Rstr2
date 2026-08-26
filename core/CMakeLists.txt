cmake_minimum_required(VERSION 3.20)
# CXX only: the OptiX kernel is compiled to PTX with nvcc via a custom command
# below, so we do NOT enable the CMake CUDA language (which would force CMake to
# locate a CUDA VS toolset that CUDA 12.5 does not provide for VS2026).
project(Rstr2Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT OPTIX_INCLUDE_DIR)
    message(FATAL_ERROR
        "Rstr2Core: OPTIX_INCLUDE_DIR is not set. Point it at the OptiX SDK "
        "include/ directory (e.g. the NVIDIA/optix-dev headers). OptiX 7+ loads "
        "optix.64.dll from the NVIDIA driver at runtime, so only headers are "
        "needed at build time.")
endif()
message(STATUS "Rstr2Core: OptiX include dir = ${OPTIX_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)

# Locate nvcc for the PTX custom command (independent of the CMake CUDA language).
find_program(NVCC nvcc REQUIRED
    PATHS "$ENV{CUDA_PATH}/bin" "$ENV{CUDA_PATH_V12_5}/bin"
    DOC "CUDA compiler used to build the OptiX PTX kernel")

# --- Compile the OptiX kernel to PTX with nvcc -------------------------------
# The kernel is compiled to PTX (architecture-independent) and loaded at
# runtime by optixModuleCreate. It must NOT be compiled into the
# executable as a normal CUDA translation unit.
set(OPTIX_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/optix_kernels.cu)
set(PTX_PATH ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels.ptx)

add_custom_command(
    OUTPUT ${PTX_PATH}
    COMMAND ${NVCC} -ptx
            -I"${OPTIX_INCLUDE_DIR}"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${OPTIX_KERNEL}"
            -o "${PTX_PATH}"
            --gpu-architecture=compute_70
            # The CI runner ships VS2026/MSVC newer than CUDA 12.5 officially
            # supports; we only emit PTX (no host object), so override the
            # host-compiler version check.
            -allow-unsupported-compiler
    DEPENDS ${OPTIX_KERNEL} ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_params.h
    COMMENT "Compiling optix_kernels.cu -> optix_kernels.ptx (nvcc)")

add_custom_target(Rstr2Core_ptx DEPENDS ${PTX_PATH})

add_executable(Rstr2Core
    src/main.cpp
    src/shared_mem.cpp
    src/renderer.cpp)

set_target_properties(Rstr2Core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)

add_dependencies(Rstr2Core Rstr2Core_ptx)

# Copy the PTX next to the executable (loaded at runtime by the renderer).
add_custom_command(TARGET Rstr2Core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PTX_PATH} $<TARGET_FILE_DIR:Rstr2Core>/optix_kernels.ptx
    COMMENT "Copying optix_kernels.ptx next to Rstr2Core.exe")

target_include_directories(Rstr2Core PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS})

# CUDA driver API (cuInit, cuMem*, ...) is provided by the NVIDIA driver's
# nvcuda.dll (the Windows CUDA driver import). We delay-load it so the --ci
# smoke run (which never touches CUDA) still starts on machines without an
# NVIDIA driver; on a real GPU host nvcuda.dll is loaded on first CUDA call.
target_link_libraries(Rstr2Core PRIVATE
    CUDA::cuda_driver
    user32
    delayimp)

if(WIN32)
    target_link_options(Rstr2Core PRIVATE "/DELAYLOAD:nvcuda.dll")
endif()

# === DXR 1.1 backend (cross-vendor) ========================================
# Built separately as Rstr2Dxr.exe. OFF by default so the OptiX CI path is
# untouched; enable with -DBUILD_DXR=ON (requires the Windows SDK dxc.exe).
option(BUILD_DXR "Build the DXR 1.1 backend (Rstr2Dxr.exe)" OFF)

if(BUILD_DXR)
    # Locate dxc inside the Windows SDK (versioned bin folder).
    file(GLOB _dxc_hints "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxc.exe")
    find_program(DXC dxc HINTS ${_dxc_hints}
        PATHS "$ENV{WindowsSdkDir}/bin" "$ENV{ProgramFiles}/Windows Kits/10/bin"
        PATH_SUFFIXES x64
        DOC "DirectXShaderCompiler (dxc) used to build the DXR kernel")

    if(NOT DXC)
        message(FATAL_ERROR
            "BUILD_DXR is ON but dxc.exe was not found. Install the Windows "
            "SDK (10.0.19041+) which ships dxc.exe, or set -DDXC=<path>.")
    endif()
    message(STATUS "Rstr2Dxr: using dxc at ${DXC}")

    set(DXR_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/src/dxr_shade.hlsl)
    set(DXR_CSO  ${CMAKE_CURRENT_BINARY_DIR}/dxr_shade.cso)

    add_custom_command(
        OUTPUT ${DXR_CSO}
        COMMAND ${DXC} -T lib_6_3 -O3 "${DXR_HLSL}" -Fo "${DXR_CSO}"
        DEPENDS ${DXR_HLSL}
        COMMENT "Compiling dxr_shade.hlsl -> dxr_shade.cso (dxc)")

    add_custom_target(Rstr2Dxr_cso DEPENDS ${DXR_CSO})

    add_executable(Rstr2Dxr
        src/dxr_main.cpp
        src/dxr_renderer.cpp
        src/shared_mem.cpp)

    set_target_properties(Rstr2Dxr PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)

    add_dependencies(Rstr2Dxr Rstr2Dxr_cso)

    target_include_directories(Rstr2Dxr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # d3d12/dxgi are linked via #pragma comment(lib) in dxr_renderer.cpp, but
    # be explicit for clarity.
    target_link_libraries(Rstr2Dxr PRIVATE d3d12 dxgi user32)

    add_custom_command(TARGET Rstr2Dxr POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DXR_CSO} $<TARGET_FILE_DIR:Rstr2Dxr>/dxr_shade.cso
        COMMENT "Copying dxr_shade.cso next to Rstr2Dxr.exe")
endif()

