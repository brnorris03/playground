# Simple cross-compilation cmake example

To build the capsule executables for both x86 and RISC-V:

```bash
cmake -G Ninja -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build
```

Resulting executables:

```
build
├── capsules
│   ├── riscv
│   │   └── vecadd.exe
│   └── x86_64
│       └── vecadd.exe
```
