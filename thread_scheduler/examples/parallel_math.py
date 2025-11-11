#!/usr/bin/env python3
"""
Parallel Math Example

Multiple threads with complex dependencies computing in parallel.
Tests maximum parallelism and synchronization with 4 cores.
"""

import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thread_scheduler import create_simulation, generate_perfetto_trace
from thread_scheduler.utils import (
    print_example_header,
    print_timeline,
    print_memory_state,
)

import sys


sim = create_simulation(num_cores=4)
dev = sim.dev  # Get device instance


@sim.thread(name="host")
def host():
    """Initialize input data."""
    return [
        dev.write("x", 10),
        dev.write("y", 20),
    ]


@sim.thread(name="worker1_subtract")
def worker1():
    """Compute difference."""
    return [
        dev.wait("x"),
        dev.wait("y"),
        dev.subtract("x", "y", "diff"),
        dev.push("diff"),
    ]


@sim.thread(name="worker2_add")
def worker2():
    """Compute sum."""
    return [
        dev.wait("x"),
        dev.wait("y"),
        dev.add("x", "y", "sum"),
        dev.push("sum"),
    ]


@sim.thread(name="worker3_multiply")
def worker3():
    """Multiply results."""
    return [
        dev.wait("sum"),
        dev.wait("diff"),
        dev.multiply("sum", "diff", "result"),
        dev.push("result"),
    ]


if __name__ == "__main__":
    print_example_header(
        title="Parallel Math Operations",
        description="Multiple threads with complex dependencies computing in parallel.",
        scheduler_info="4 cores (maximum parallelism)",
    )

    events = sim.run()
    print_timeline(events, "Parallel Math Example", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_parallel_math.json")

    print_memory_state(
        sim.get_memory(),
        variables=[
            ("x", None),
            ("y", None),
            ("diff", -10),
            ("sum", 30),
            ("result", -300),
        ],
    )
