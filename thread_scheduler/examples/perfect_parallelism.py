#!/usr/bin/env python3
"""
Perfect Parallelism Demo

Demonstrates optimal parallel execution achieving 100% efficiency on 4 cores.
Four independent workers execute simultaneously with no dependencies or blocking.
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
dev = sim.dev  # Get device instance


@sim.thread(name="worker1", use_ast=False)
def worker1():
    """Compute result1 independently."""
    return [
        dev.write("a", 10),
        dev.push("a"),
        dev.write("b", 20),
        dev.push("b"),
        dev.add("a", "b", "result1"),
        dev.push("result1"),
    ]


@sim.thread(name="worker2", use_ast=False)
def worker2():
    """Compute result2 independently."""
    return [
        dev.write("c", 30),
        dev.push("c"),
        dev.write("d", 40),
        dev.push("d"),
        dev.add("c", "d", "result2"),
        dev.push("result2"),
    ]


@sim.thread(name="worker3", use_ast=False)
def worker3():
    """Compute result3 independently."""
    return [
        dev.write("e", 50),
        dev.push("e"),
        dev.write("f", 60),
        dev.push("f"),
        dev.add("e", "f", "result3"),
        dev.push("result3"),
    ]


@sim.thread(name="worker4", use_ast=False)
def worker4():
    """Compute result4 independently."""
    return [
        dev.write("g", 70),
        dev.push("g"),
        dev.write("h", 80),
        dev.push("h"),
        dev.add("g", "h", "result4"),
        dev.push("result4"),
    ]


if __name__ == "__main__":
    print_example_header(
        title="Perfect Parallelism",
        description="Four independent workers achieving 100% efficiency on 4 cores",
        scheduler_info="4 cores",
    )

    events = sim.run()
    print_timeline(events, "Perfect Parallelism", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_perfect_parallelism.json")

    print_memory_state(
        sim.get_memory(),
        variables=[
            ("result1", 30),
            ("result2", 70),
            ("result3", 110),
            ("result4", 150),
        ],
    )
