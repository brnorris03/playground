cmake_minimum_required(VERSION 3.21)

project(Example LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_RISCV_COMPILATION)
    set(BINARY_SUBDIR riscv)
else()
    set(BINARY_SUBDIR x86)
endif()

add_executable(vecadd.exe vecadd_main.c vecadd.c)
