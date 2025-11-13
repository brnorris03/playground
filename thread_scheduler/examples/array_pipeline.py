#!/usr/bin/env python3
"""
Array Pipeline Example

Three-stage pipeline (read, compute, write) operating on large arrays.
Demonstrates streaming computation with 1024-element chunks over 10 iterations.
Each iteration launches a set of 3 threads (reader, compute, writer).
"""

import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thread_scheduler import create_simulation, generate_perfetto_trace
from thread_scheduler.utils import (
    print_example_header,
    print_timeline,
)


sim = create_simulation(num_cores=3)
dev = sim.dev

# Configuration
CHUNK_SIZE = 1024
NUM_ITERATIONS = 10
TOTAL_SIZE = CHUNK_SIZE * NUM_ITERATIONS


# Define the device program as a set of thread functions
def device_program(iteration):
    """Define the 3-thread pipeline for a given iteration."""

    @sim.thread(name=f"It{iteration}_reader", use_ast=False)
    def reader():
        """Read input chunk."""
        # Simulate reading array chunks with numeric values
        return [
            dev.write(f"a_chunk_{iteration}", iteration * 10),
            dev.push(f"a_chunk_{iteration}"),
            dev.write(f"b_chunk_{iteration}", iteration * 5),
            dev.push(f"b_chunk_{iteration}"),
        ]

    @sim.thread(name=f"It{iteration}_compute", use_ast=False)
    def compute():
        """Compute on chunk: result = (a + b) * (a - b)."""
        return [
            dev.wait(f"a_chunk_{iteration}"),
            dev.wait(f"b_chunk_{iteration}"),
            dev.add(
                f"a_chunk_{iteration}", f"b_chunk_{iteration}", f"sum_chunk_{iteration}"
            ),
            dev.subtract(
                f"a_chunk_{iteration}",
                f"b_chunk_{iteration}",
                f"diff_chunk_{iteration}",
            ),
            dev.multiply(
                f"sum_chunk_{iteration}",
                f"diff_chunk_{iteration}",
                f"result_chunk_{iteration}",
            ),
            dev.push(f"result_chunk_{iteration}"),
        ]

    @sim.thread(name=f"It{iteration}_writer", use_ast=False)
    def writer():
        """Write output chunk."""
        return [
            dev.wait(f"result_chunk_{iteration}"),
            dev.write(f"output_{iteration}", f"result_chunk_{iteration}"),
            dev.push(f"output_{iteration}"),
        ]


# Launch the device program for each iteration
for i in range(NUM_ITERATIONS):
    device_program(i)


if __name__ == "__main__":
    print_example_header(
        title="Array Pipeline",
        description=f"Three-stage pipeline processing {TOTAL_SIZE} elements in {CHUNK_SIZE}-element chunks.",
        scheduler_info="3 cores",
    )

    events = sim.run()
    print_timeline(events, "Array Pipeline", num_cores=sim.get_num_cores())
    generate_perfetto_trace(
        events, "trace_array_pipeline.json", num_cores=sim.get_num_cores()
    )

    print("\nPipeline Statistics:")
    print("=" * 80)
    print(f"Array size: {TOTAL_SIZE} elements")
    print(f"Chunk size: {CHUNK_SIZE} elements")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Threads: {NUM_ITERATIONS * 3} (reader, compute, writer per chunk)")
    print(f"Operations per chunk:")
    print(f"  Reader: 4 ops (2 writes, 2 pushes)")
    print(f"  Compute: 6 ops (2 waits, 3 math ops, 1 push)")
    print(f"  Writer: 3 ops (1 wait, 1 write, 1 push)")
    print(f"Total operations: {NUM_ITERATIONS * 13}")
    print("=" * 80 + "\n")
