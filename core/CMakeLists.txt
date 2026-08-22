cmake_minimum_required(VERSION 3.20)
project(Rstr2Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Rstr2Core src/main.cpp)

# Phase 2 will add: d3d12, dxgi, dxguid libs + HLSL->DXIL runtime compile via dxc.
if(WIN32)
    target_link_libraries(Rstr2Core PRIVATE user32)
endif()
