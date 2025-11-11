#!/usr/bin/env python3
"""
Producer-Consumer Example

Simple synchronization between a producer and consumer thread.
Tests basic write/wait operations.
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


sim = create_simulation(num_cores=2)
dev = sim.dev  # Get device instance


@sim.thread(name="producer")
def producer():
    """Produce data."""
    return [
        dev.write("data", 42),
    ]


@sim.thread(name="consumer")
def consumer():
    """Wait for and consume data."""
    return [
        dev.wait("data"),
        dev.write("processed", 100),
    ]


if __name__ == "__main__":
    print_example_header(
        title="Producer-Consumer",
        description="Simple synchronization between producer and consumer threads.",
        scheduler_info="2 cores",
    )

    events = sim.run()
    print_timeline(events, "Producer-Consumer Example", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_producer_consumer.json")

    print_memory_state(sim.get_memory())
