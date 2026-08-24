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
