cmake_minimum_required(VERSION 3.20)
project(Rstr2Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.hlsl)
set(CSO_PATH ${CMAKE_CURRENT_BINARY_DIR}/raytracing.cso)

# Locate dxc (DirectX Shader Compiler). Prefer an explicitly provided path
# (DXC_EXECUTABLE env var, set by CI to a pinned DirectXShaderCompiler build
# because the Windows SDK dxc on some runners does not expose DXR builtins),
# then fall back to a Windows SDK glob and finally a PATH search.
if(DEFINED ENV{DXC_EXECUTABLE} AND NOT "$ENV{DXC_EXECUTABLE}" STREQUAL "")
    set(DXC_EXECUTABLE "$ENV{DXC_EXECUTABLE}")
endif()
if(NOT DXC_EXECUTABLE)
    file(GLOB _dxc_list
        "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxc.exe"
        "C:/Program Files/Windows Kits/10/bin/*/x64/dxc.exe")
    if(_dxc_list)
        list(GET _dxc_list 0 DXC_EXECUTABLE)
    endif()
endif()
if(NOT DXC_EXECUTABLE)
    find_program(DXC_EXECUTABLE dxc)
endif()

if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR
        "Rstr2Core: dxc (DirectX Shader Compiler) not found. Install the Windows "
        "SDK (10.x) or add dxc.exe to PATH so shaders/raytracing.hlsl can be "
        "compiled to raytracing.cso.")
endif()

# If a prebuilt .cso was supplied (RAYTRACING_CSO env, set by CI's dxc fetch
# step where the compiler DLLs are guaranteed adjacent and verified), copy it.
# Otherwise compile raytracing.hlsl with dxc.
if(DEFINED ENV{RAYTRACING_CSO} AND NOT "$ENV{RAYTRACING_CSO}" STREQUAL "")
    set(RAYTRACING_CSO "$ENV{RAYTRACING_CSO}")
    message(STATUS "Rstr2Core: using prebuilt cso at ${RAYTRACING_CSO}")
    add_custom_command(
        OUTPUT ${CSO_PATH}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RAYTRACING_CSO}" "${CSO_PATH}"
        DEPENDS ${RAYTRACING_CSO}
        COMMENT "Copying prebuilt raytracing.cso")
else()
    if(NOT DXC_EXECUTABLE)
        message(FATAL_ERROR
            "Rstr2Core: dxc (DirectX Shader Compiler) not found. Install the Windows "
            "SDK (10.x) or add dxc.exe to PATH so shaders/raytracing.hlsl can be "
            "compiled to raytracing.cso.")
    endif()
    message(STATUS "Rstr2Core: using dxc at ${DXC_EXECUTABLE}")
    add_custom_command(
        OUTPUT ${CSO_PATH}
        COMMAND ${DXC_EXECUTABLE} -T lib_6_5 -HV 2021 -Fo ${CSO_PATH} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling raytracing.hlsl -> raytracing.cso (dxc)")
endif()

# The .cso is loaded at runtime from the executable directory.
#
# NOTE: raytracing.hlsl must NOT be listed as a target source. Otherwise
# MSBuild/Visual Studio compiles it as a shader with its default (non-DXR)
# rules and fails on RaytracingAccelerationStructure. It is only a DEPENDS of
# the custom command below; we remove it from the sources here.
add_executable(Rstr2Core
    src/main.cpp
    src/shared_mem.cpp
    src/renderer.cpp)

set_target_properties(Rstr2Core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)

# Compile the shader to raytracing.cso, then copy it next to the exe.
add_custom_target(Rstr2Core_cso DEPENDS ${CSO_PATH})
add_dependencies(Rstr2Core Rstr2Core_cso)

add_custom_command(TARGET Rstr2Core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CSO_PATH} $<TARGET_FILE_DIR:Rstr2Core>/raytracing.cso
    COMMENT "Copying raytracing.cso next to Rstr2Core.exe")

# --- Optional Agility SDK (matching D3D12 debug layer) ---
# When AGILITY_SDK_DIR points at an extracted Microsoft.Direct3D.D3D12
# package's build/native directory, use its headers (so D3D12_AGILITY_SDK_VERSION
# is defined) and ship its D3D12Core.dll + d3d12SDKLayers.dll next to the exe.
# The OS loader then side-loads the Agility runtime + a version-matched debug
# layer, which is required to get precise D3D12 validation errors (e.g. from
# CreateStateObject) on systems whose OS d3d12.dll / Graphics-Tools debug layer
# are version-mismatched. The app still links the normal Windows SDK d3d12.lib.
if(DEFINED ENV{AGILITY_SDK_DIR} AND NOT "$ENV{AGILITY_SDK_DIR}" STREQUAL "")
    set(AGILITY_SDK_DIR "$ENV{AGILITY_SDK_DIR}")
    message(STATUS "Rstr2Core: using Agility SDK at ${AGILITY_SDK_DIR}")
    target_include_directories(Rstr2Core PRIVATE
        ${AGILITY_SDK_DIR}/include
        ${AGILITY_SDK_DIR}/include/d3dx12)
    add_custom_command(TARGET Rstr2Core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${AGILITY_SDK_DIR}/bin/x64/D3D12Core.dll
                $<TARGET_FILE_DIR:Rstr2Core>/D3D12Core.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${AGILITY_SDK_DIR}/bin/x64/d3d12SDKLayers.dll
                $<TARGET_FILE_DIR:Rstr2Core>/d3d12SDKLayers.dll
        COMMENT "Copying Agility SDK D3D12Core.dll + d3d12SDKLayers.dll next to Rstr2Core.exe")
endif()

if(WIN32)
    target_link_libraries(Rstr2Core PRIVATE user32 d3d12 dxgi dxguid)
    # DXR driver call chains (notably CreateStateObject / device init) can
    # exceed the 1 MB default thread stack and crash with STATUS_STACK_OVERFLOW
    # (0xC00000FD) on some drivers/GPUs. Reserve a larger stack up front.
    target_link_options(Rstr2Core PRIVATE "/STACK:16777216")
endif()
