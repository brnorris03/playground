# Performance profiling

Th
## Profiling

To profile kernel execution, run with the device profiler enabled:

```bash
TT_METAL_DEVICE_PROFILER=1 pytest -svx test_fused_kernel.py -k "64"
```

When `TT_METAL_DEVICE_PROFILER=1` is set:
1. The JIT build system adds `-DPROFILE_KERNEL` to kernel compiles, enabling profiler macros.
2. `DeviceZoneScopedN("zone-name")` in kernel code records timing for that scope.
3. Profiler data is written to `generated/profiler/.logs/profile_log_device.csv`.

Post-process results with:
```bash
$TT_METAL_ROOT/tools/tracy/process_device_log.py
```

This generates `generated/profiler/output/device/device_analysis_data.json` with timing statistics.

The compute kernel includes a `DeviceZoneScopedN("COMPUTE-block")` zone that measures block processing time. Without `TT_METAL_DEVICE_PROFILER=1`, the macro compiles to nothing (zero overhead).

### Example Profiling Output

```
2025-12-07 22:32:20.098 | INFO     | __main__:core_to_device_timeseries:414 - Importing Data For Device Number : 0

=================== device_kernel_first_to_last_start ===================
    Duration [cycles] =        317


=================== device_kernel_duration_per_core ===================
       Count          =         64
     Average [cycles] =      4,587
         Max [cycles] =      5,324
      Median [cycles] =      4,620
         Min [cycles] =      3,804
         Sum [cycles] =    293,540
       Range [cycles] =      1,520


=================== device_fw_duration ===================
    Duration [cycles] =      6,075


=================== device_kernel_duration ===================
    Duration [cycles] =      5,372


=================== device_kernel_duration_dm_start ===================
    Duration [cycles] =      5,288


=================== device_brisc_kernel_duration ===================
    Duration [cycles] =      5,211


=================== device_ncrisc_kernel_duration ===================
    Duration [cycles] =      2,557


=================== device_trisc0_kernel_duration ===================
    Duration [cycles] =      2,886


=================== device_trisc1_kernel_duration ===================
    Duration [cycles] =      4,772


=================== device_trisc2_kernel_duration ===================
    Duration [cycles] =        494
```
### Analysis of Profiling Results

Your Tracy profile shows a kernel running on 64 cores with moderate performance variance. The total execution takes 5,372 cycles, with TRISC2 (packer) being the bottleneck at only 494 cycles, indicating it's waiting for other components.

## Detailed Interpretation

### Overall Timing Metrics

| Metric | Cycles | Interpretation |
|--------|--------|----------------|
| **device_fw_duration** | 6,075 | Total firmware overhead including kernel setup/teardown |
| **device_kernel_duration** | 5,372 | Actual kernel execution time (first to last core) |
| **device_kernel_duration_dm_start** | 5,288 | Kernel time from data movement kernel start |

The firmware overhead (6,075 - 5,372 = 703 cycles) represents ~13% of total time, which is reasonable.

### RISC Processor Breakdown

| Processor | Cycles | Role |
|-----------|--------|------|
| **BRISC** | 5,211 | Data movement (reader) |
| **NCRISC** | 2,557 | Data movement (writer) |
| **TRISC0** | 2,886 | Unpacker |
| **TRISC1** | 4,772 | Math computation |
| **TRISC2** | 494 | Packer |

The profiling shows BRISC (reader) at 5,211 cycles is the bottleneck, not TRISC1 (math) at 4,772 cycles. This means we're read-bound. 

## Possible optimizations
1. Double-Buffering (implemented in the *_db.cpp kernel versions)
- Overlap reading the next block with computing the current block:
- Double CB size: 128 tiles (2 blocks) instead of 64 tiles (1 block)
- Reader prefetches next block while compute processes current
- Potential speedup: ~2x for read-bound workloads
- Trade-off: Uses 2x L1 memory for input CBs (384KB → 768KB for 3 inputs)

2. Reduce Block Size
- Smaller blocks = more overlap opportunities, less memory
- Current: 64 tiles/block (8×8)
- Could try: 16 tiles/block (4×4) or 32 tiles/block
- Trade-off: More block overhead, less efficient NOC utilization

3. Defer C Read
- Since C is only needed for the final add, delay reading C until after exp():
- Read A, B first → compute A+B, exp()
- Then read C → compute + C
- Reduces peak CB memory pressure

## Double-Buffering Results (Single Core)

### Summary - EXP Operation

| Tiles | Blocks | Single-buf (cycles) | Double-buf (cycles) | Improvement |
|-------|--------|---------------------|---------------------|-------------|
| 128   | 2      | 271,712             | 264,421             | 2.7%        |
| 256   | 4      | 538,585             | 516,853             | 4.0%        |
| 1024  | 16     | 2,141,663           | 2,031,266           | 5.2%        |

### Detailed Results (1024 tiles = 16 blocks) - EXP

**Single-Buffered:**
| Processor | Cycles | Role |
|-----------|--------|------|
| **BRISC** | 2,141,474 | Reader |
| **NCRISC** | 2,011,398 | Writer |
| **TRISC1** | 2,137,737 | Math |

**Double-Buffered:**
| Processor | Cycles | Role |
|-----------|--------|------|
| **BRISC** | 2,031,078 | Reader |
| **NCRISC** | 1,781,942 | Writer |
| **TRISC1** | 2,027,335 | Math |

### Analysis - EXP Operation

The modest 5% improvement even with 16 blocks indicates this kernel is **compute-bound**:

- **TRISC1 (math) dominates** at ~2M cycles - this is the bottleneck
- BRISC (reader) and TRISC1 (math) have similar cycle counts, but the reader is limited by how fast compute can consume data
- Double-buffering helps overlap read/compute, but when compute is the bottleneck, there's limited benefit

**Why is this kernel compute-bound?**

The fused `exp(A + B) + C` operation involves:
1. Two `add_binary_tile` operations
2. One `exp_tile` operation (expensive transcendental)
3. Multiple `copy_tile` operations

The `exp_tile` SFPU operation is expensive, making compute the bottleneck rather than DRAM reads.

---

## Double-Buffering Results with ReLU (Single Core, 128 tiles = 2 blocks)

ReLU is a cheap operation compared to EXP, so we expect different bottleneck behavior.

### Summary - ReLU Operation

| Config | Single-buf (cycles) | Double-buf (cycles) | Improvement |
|--------|---------------------|---------------------|-------------|
| 128 tiles, single core | 114,092 | 106,739 | 6.3% |

### Detailed Comparison (128 tiles = 2 blocks) - ReLU

**Single-Buffered:**
| Processor | Cycles | Role |
|-----------|--------|------|
| **BRISC** | 113,870 | Reader |
| **NCRISC** | 62,605 | Writer |
| **TRISC0** | 107,471 | Unpack |
| **TRISC1** | 110,136 | Math |
| **TRISC2** | 110,067 | Pack |

**Double-Buffered:**
| Processor | Cycles | Role |
|-----------|--------|------|
| **BRISC** | 106,518 | Reader |
| **NCRISC** | 15,637 | Writer |
| **TRISC0** | 100,159 | Unpack |
| **TRISC1** | 102,773 | Math |
| **TRISC2** | 102,733 | Pack |

### Analysis - ReLU Operation

Double buffering shows **consistent 6-7% improvements** across all compute threads, with the **Writer showing a notable 75% improvement**:

- **BRISC (Reader)**: 6.4% faster (113,870 → 106,518)
- **NCRISC (Writer)**: **75% faster** (62,605 → 15,637)
- **TRISC0 (Unpack)**: 6.8% faster (107,471 → 100,159)
- **TRISC1 (Math)**: 6.7% faster (110,136 → 102,773)
- **TRISC2 (Pack)**: 6.6% faster (110,067 → 102,733)

**Why ReLU shows consistent improvements:**

Reader and math are balanced (106,518 vs 102,773 cycles), with neither clearly dominant:
1. The 2x CB size enables better pipelining: reader prefetches block N+1 while compute processes block N
2. All threads benefit from reduced waiting on the alternate buffer
3. The writer's dramatic 75% improvement suggests it was experiencing more stalls without double buffering, but this appears to be a synchronization artifact rather than indicating a fundamentally different bottleneck

**Comparison: EXP vs ReLU with Double Buffering**

| Operation | Overall Improvement | Reader vs Math | Observation |
|-----------|---------------------|---|---|
| EXP (1024 tiles) | 5.2% | Balanced (2,031M vs 2,027M) | Conservative improvements across all threads |
| ReLU (128 tiles) | 6.3% | Balanced (106.5k vs 102.8k) | Similar balance; writer stall reduction is outlier |

Both workloads show reader and math approximately balanced, suggesting **double buffering's benefit comes from reduced synchronization stalls and better CB occupancy**, not from shifting a compute-bound bottleneck to become memory-bound.
