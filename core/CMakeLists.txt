cmake_minimum_required(VERSION 3.20)
project(Rstr2Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.hlsl)
set(CSO_PATH ${CMAKE_CURRENT_BINARY_DIR}/raytracing.cso)
set(CSO_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/raytracing_cso.h)

# Tell the Visual Studio generator to also recognize/compile the HLSL as a
# shader library (per the Phase 2 spec). The object file name is intentionally
# distinct from CSO_PATH so it never collides with the dxc-generated cso below.
set_source_files_properties(${SHADER_SRC} PROPERTIES
    VS_SHADER_TYPE "Library"
    VS_SHADER_MODEL "6.3"
    VS_SHADER_ENTRY_POINT ""
    VS_SHADER_OBJECT_FILE_NAME "raytracing_vs.cso")

# Locate dxc (DirectX Shader Compiler). The Windows SDK ships it under a
# versioned bin folder, so glob for it in addition to a PATH search.
set(DXC_EXECUTABLE "")
file(GLOB _dxc_list
    "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxc.exe"
    "C:/Program Files/Windows Kits/10/bin/*/x64/dxc.exe")
if(_dxc_list)
    list(GET _dxc_list 0 DXC_EXECUTABLE)
endif()
if(NOT DXC_EXECUTABLE)
    find_program(DXC_EXECUTABLE dxc)
endif()

if(DXC_EXECUTABLE)
    message(STATUS "Rstr2Core: using dxc at ${DXC_EXECUTABLE}")
    add_custom_command(
        OUTPUT ${CSO_PATH}
        COMMAND ${DXC_EXECUTABLE} -T lib_6_3 -Fo ${CSO_PATH} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling raytracing.hlsl -> raytracing.cso (dxc)")
    add_custom_command(
        OUTPUT ${CSO_HEADER}
        COMMAND ${CMAKE_COMMAND} -DINPUT=${CSO_PATH} -DOUTPUT=${CSO_HEADER}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateCsoHeader.cmake
        DEPENDS ${CSO_PATH}
        COMMENT "Generating raytracing_cso.h from cso")
else()
    message(WARNING
        "Rstr2Core: dxc not found - generating a STUB raytracing_cso.h. "
        "The renderer will report a clear error at runtime if DXR is attempted; "
        "the CI --ci path is unaffected.")
    add_custom_command(
        OUTPUT ${CSO_HEADER}
        COMMAND ${CMAKE_COMMAND} -DOUTPUT=${CSO_HEADER}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateCsoHeader.cmake
        DEPENDS ${SHADER_SRC}
        COMMENT "Generating stub raytracing_cso.h (dxc not found)")
endif()

add_executable(Rstr2Core
    src/main.cpp
    src/shared_mem.cpp
    src/renderer.cpp
    src/raytracing_cso.h
    ${SHADER_SRC})

# Make sure the generated header exists before compiling the target.
add_custom_target(Rstr2Core_cso_header DEPENDS ${CSO_HEADER})
add_dependencies(Rstr2Core Rstr2Core_cso_header)

if(WIN32)
    target_link_libraries(Rstr2Core PRIVATE user32 d3d12 dxgi dxguid)
endif()
