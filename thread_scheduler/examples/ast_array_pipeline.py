#!/usr/bin/env python3
"""
AST Array Pipeline Example

Three-stage pipeline (read, compute, write) using natural Python syntax with iteration scoping.
Demonstrates streaming computation with 1024-element chunks over 10 iterations.
Each iteration launches a set of 3 threads (reader, compute, writer) with iteration-scoped variables.
"""

import sys
import random
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thread_scheduler import create_simulation, generate_perfetto_trace, read, store
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

# Set random seed for reproducibility
random.seed(42)

# Generate random input values for each iteration
input_values = [
    (random.randint(1, 100), random.randint(1, 100)) for _ in range(NUM_ITERATIONS)
]


# Define the device program as a set of thread functions
def device_program(iteration):
    """Define the 3-thread pipeline for a given iteration using AST syntax with iteration scoping."""

    # Get the random values for this iteration
    a_val, b_val = input_values[iteration]

    @sim.thread(name=f"It{iteration}_reader", iteration=iteration)
    def reader():
        """Read input chunk."""
        a = read(a_val)
        b = read(b_val)
        return a, b

    @sim.thread(name=f"It{iteration}_compute", iteration=iteration)
    def compute():
        """Compute on chunk: result = (a + b) * (a - b)."""
        sum_val = a + b
        diff_val = a - b
        result = sum_val * diff_val
        return result

    @sim.thread(name=f"It{iteration}_writer", iteration=iteration)
    def writer():
        """Write output chunk."""
        store(result, f"output_{iteration}")
        return None


if __name__ == "__main__":
    print_example_header(
        title="AST Array Pipeline",
        description=f"Three-stage pipeline processing {TOTAL_SIZE} elements in {CHUNK_SIZE}-element chunks.",
        scheduler_info="3 cores, iteration-scoped variables",
    )

    # Launch the device program for each iteration
    for i in range(NUM_ITERATIONS):
        device_program(i)

    events = sim.run()
    print_timeline(events, "AST Array Pipeline", num_cores=sim.get_num_cores())
    generate_perfetto_trace(
        events, "trace_ast_array_pipeline.json", num_cores=sim.get_num_cores()
    )

    print("\nPipeline Statistics:")
    print("=" * 80)
    print(f"Array size: {TOTAL_SIZE} elements")
    print(f"Chunk size: {CHUNK_SIZE} elements")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Threads: {NUM_ITERATIONS * 3} (reader, compute, writer per chunk)")
    print(f"Operations per iteration:")
    print(f"  Reader: 4 ops (2 writes, 2 pushes)")
    print(f"  Compute: 9 ops (2 waits, 2 math ops, 1 push for each intermediate)")
    print(f"  Writer: 2 ops (1 wait, 1 assignment)")
    print(f"Total operations: ~{NUM_ITERATIONS * 15}")
    print("=" * 80 + "\n")
