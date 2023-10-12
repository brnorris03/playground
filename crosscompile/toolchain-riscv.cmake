# the name of the target operating system
set(CMAKE_SYSTEM_NAME CapsuleOs)
set(CMAKE_SYSTEM_VERSION 1.0)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-elf-g++)

# where is the target environment located
set(CMAKE_FIND_ROOT_PATH /home/norris/soft/riscv-gnu-toolchain)

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

SET(CMAKE_RISCV_COMPILATION 1)
