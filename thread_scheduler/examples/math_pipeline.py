#!/usr/bin/env python3
"""
Math Pipeline Example

Pipeline of math operations with dependencies.
Tests compute-then-push pattern and sequential dependencies.
"""

import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thread_scheduler import create_simulation, generate_perfetto_trace
from thread_scheduler.utils import print_example_header, print_timeline, print_memory_state

import sys


sim = create_simulation(num_cores=3)
dev = sim.dev  # Get device instance


@sim.thread(name="initializer")
def initializer():
    """Initialize input values."""
    return [
        dev.write("a", 5),
        dev.write("b", 3),
    ]


@sim.thread(name="adder")
def adder():
    """Compute sum."""
    return [
        dev.wait("a"),
        dev.wait("b"),
        dev.add("a", "b", "sum"),
        dev.push("sum"),
    ]


@sim.thread(name="multiplier")
def multiplier():
    """Compute product."""
    return [
        dev.wait("sum"),
        dev.write("c", 2),
        dev.multiply("sum", "c", "product"),
        dev.push("product"),
    ]


if __name__ == "__main__":
    print_example_header(
        example_num=3,
        title="Math Pipeline",
        description="Pipeline of math operations with sequential dependencies.",
        scheduler_info="3 cores",
    )

    events = sim.run()
    print_timeline(events, "Math Pipeline Example", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_math_pipeline.json")

    print_memory_state(
        sim.get_memory(),
        variables=[
            ("a", None),
            ("b", None),
            ("sum", 8),
            ("c", None),
            ("product", 16),
        ],
    )
