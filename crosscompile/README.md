# Simple cross-compilation cmake example

### Building with a toolchain different from the host one

Configure using the toolchain file

```bash
cmake -G Ninja -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_TOOLCHAIN_FILE=toolchain-riscv.cmake -B build-riscv
cmake --build build-riscv
```

To build using the same compiler options as the host system, simply omit the `DCMAKE_BUILD_TYPE` argument:

```bash
cmake -G Ninja -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_BUILD_TYPE=Debug \
	-B build
cmake --build build
```
