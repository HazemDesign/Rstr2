cmake_minimum_required(VERSION 3.20)
project(Rstr2Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.hlsl)
set(CSO_PATH ${CMAKE_CURRENT_BINARY_DIR}/raytracing.cso)

# Locate dxc (DirectX Shader Compiler). The Windows SDK ships it under a
# versioned bin folder, so glob fortran... glob for it.
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

if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR
        "Rstr2Core: dxc (DirectX Shader Compiler) not found. Install the Windows "
        "SDK (10.x) or add dxc.exe to PATH so shaders/raytracing.hlsl can be "
        "compiled to raytracing.cso.")
endif()

message(STATUS "Rstr2Core: using dxc at ${DXC_EXECUTABLE}")
add_custom_command(
    OUTPUT ${CSO_PATH}
    COMMAND ${DXC_EXECUTABLE} -T lib_6_3 -Fo ${CSO_PATH} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling raytracing.hlsl -> raytracing.cso (dxc)")

# The .cso is loaded at runtime from the executable directory.
set_target_properties(Rstr2Core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)

add_executable(Rstr2Core
    src/main.cpp
    src/shared_mem.cpp
    src/renderer.cpp
    ${SHADER_SRC})

# Compile the shader to raytracing.cso, then copy it next to the exe.
add_custom_target(Rstr2Core_cso DEPENDS ${CSO_PATH})
add_dependencies(Rstr2Core Rstr2Core_cso)

add_custom_command(TARGET Rstr2Core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CSO_PATH} $<TARGET_FILE_DIR:Rstr2Core>/raytracing.cso
    COMMENT "Copying raytracing.cso next to Rstr2Core.exe")

if(WIN32)
    target_link_libraries(Rstr2Core PRIVATE user32 d3d12 dxgi dxguid)
endif()
