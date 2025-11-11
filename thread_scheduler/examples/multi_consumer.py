#!/usr/bin/env python3
"""
Multi-Consumer Example

One producer writes data, multiple consumers wait and process it.
Tests parallel consumer execution.
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


@sim.thread(name="producer")
def producer():
    """Produce data for consumers."""
    return [
        dev.write("data", 100),
    ]


@sim.thread(name="consumer1")
def consumer1():
    """First consumer."""
    return [
        dev.wait("data"),
        dev.write("result1", 200),
    ]


@sim.thread(name="consumer2")
def consumer2():
    """Second consumer."""
    return [
        dev.wait("data"),
        dev.write("result2", 300),
    ]


if __name__ == "__main__":
    print_example_header(
        example_num=2,
        title="Multi-Consumer Pattern",
        description="One producer, multiple consumers processing in parallel.",
        scheduler_info="3 cores (all threads can run in parallel)",
    )

    events = sim.run()
    print_timeline(events, "Multi-Consumer Example", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_multi_consumer.json")

    print_memory_state(
        sim.get_memory(),
        variables=[
            ("data", None),
            ("result1", None),
            ("result2", None),
        ],
    )
