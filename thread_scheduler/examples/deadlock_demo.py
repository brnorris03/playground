#!/usr/bin/env python3
"""
Deadlock Detection Demo

Demonstrates the simulator's deadlock detection capability.
Two threads wait for data that will never be written, causing a deadlock.
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


@sim.thread(name="worker1", use_ast=False)
def worker1():
    """Wait for data that worker2 will provide (but worker2 is also waiting)."""
    return [
        dev.wait("data_from_worker2"),
        dev.write("data_from_worker1", 100),
        dev.push("data_from_worker1"),
    ]


@sim.thread(name="worker2", use_ast=False)
def worker2():
    """Wait for data that worker1 will provide (but worker1 is also waiting)."""
    return [
        dev.wait("data_from_worker1"),
        dev.write("data_from_worker2", 200),
        dev.push("data_from_worker2"),
    ]


if __name__ == "__main__":
    print_example_header(
        title="Deadlock Detection Demo",
        description="Two threads waiting for each other - demonstrates deadlock detection",
        scheduler_info="2 cores",
    )

    events = sim.run()
    print_timeline(events, "Deadlock Demo", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_deadlock_demo.json")

    print_memory_state(sim.get_memory())

    print(
        f"""
{'='*80}
{'DEADLOCK EXPLANATION':^80}
{'='*80}

What happened:
  - worker1 is waiting for 'data_from_worker2'
  - worker2 is waiting for 'data_from_worker1'
  - Neither can proceed because both are blocked
  - The simulator detected this deadlock and issued a warning

Note: The simulator continues execution but no progress can be made.
{'='*80}
"""
    )
