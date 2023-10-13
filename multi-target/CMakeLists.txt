cmake_minimum_required(VERSION 3.21)

project(Example LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CAPSULE_SUBIDR ${CMAKE_BINARY_DIR}/capsules)

function(compile_capsule capsule_name device)
    set(CAPSULE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${capsule_name}.c ${CMAKE_CURRENT_SOURCE_DIR}/${capsule_name}_main.c)

    # Can be in a function
    if(device STREQUAL "riscv")
        set(DEVICE_C_COMPILER "$ENV{RISCV}/bin/riscv64-unknown-elf-gcc")
        set(DEVICE_C_FLAGS ${CMAKE_C_FLAGS} -march=rv64imafc -mabi=lp64)
    else()
        set(DEVICE_C_COMPILER "gcc")
        set(DEVICE_C_FLAGS ${CMAKE_C_FLAGS} -march=native)
    endif()

    add_custom_command(
        OUTPUT ${CAPSULE_SUBIDR}/${device}/${capsule_name}.exe
        COMMAND ${DEVICE_C_COMPILER} ${DEVICE_C_FLAGS} -o ${CAPSULE_SUBIDR}/${device}/${capsule_name}.exe ${CAPSULE_SOURCES}
        DEPENDS ${CAPSULE_SOURCES}
    )
endfunction()

foreach(DEVICE "x86_64" "riscv")
    compile_capsule(vecadd ${DEVICE})
endforeach()

add_custom_target(
    capsules ALL
    DEPENDS ${CAPSULE_SUBIDR}/x86_64/vecadd.exe ${CAPSULE_SUBIDR}/riscv/vecadd.exe
)
