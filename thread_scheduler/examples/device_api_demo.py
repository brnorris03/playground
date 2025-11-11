#!/usr/bin/env python3
"""
Device API Demo

Demonstrates the clean Device API for creating operations.
No need to call create_operation - just use dev.write(), dev.add(), etc.
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


sim = create_simulation(num_cores=4)
dev = sim.dev  # Get the device instance


@sim.thread(name="host")
def host():
    """Initialize input data using Device API."""
    return [
        dev.write("x", 10),
        dev.write("y", 20),
    ]


@sim.thread(name="worker1")
def worker1():
    """Compute difference using Device API."""
    return [
        dev.wait("x"),
        dev.wait("y"),
        dev.subtract("x", "y", "diff"),
        dev.push("diff"),
    ]


@sim.thread(name="worker2")
def worker2():
    """Compute sum using Device API."""
    return [
        dev.wait("x"),
        dev.wait("y"),
        dev.add("x", "y", "sum"),
        dev.push("sum"),
    ]


@sim.thread(name="worker3")
def worker3():
    """Multiply results using Device API."""
    return [
        dev.wait("sum"),
        dev.wait("diff"),
        dev.multiply("sum", "diff", "result"),
        dev.push("result"),
    ]


if __name__ == "__main__":
    print_example_header(
        example_num=0,
        title="Device API Demo",
        description="Clean API using dev.write(), dev.add(), dev.push(), etc.",
        scheduler_info="4 cores",
    )

    events = sim.run()
    print_timeline(events, "Device API Demo", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_device_api_demo.json")

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
