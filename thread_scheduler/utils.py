#!/usr/bin/env python3
"""
Utility functions for the thread scheduler simulator.
"""

from typing import Dict, List, Tuple, Optional


def print_example_header(example_num, title, description, scheduler_info):
    """Print a standardized header for examples."""
    print("\n" + "=" * 80)
    print(f"EXAMPLE {example_num}: {title}")
    print("=" * 80)
    print(f"Description: {description}")
    print(f"Scheduler: {scheduler_info}")
    print()


def print_timeline(events, title="Execution Timeline", num_cores=None):
    """
    Print a console timeline summary with efficiency statistics.

    Args:
        events: List of execution events
        title: Title for the timeline
        num_cores: Number of cores (if provided, shows efficiency stats)
    """
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"{'Time':<10} {'Thread':<20} {'Operation':<30} {'Status':<15}")
    print(f"{'-'*80}")

    for event in events:
        time_str = f"{event.start_time:.1f}s"
        if event.status == "completed":
            time_str += f" - {event.end_time:.1f}s"

        print(
            f"{time_str:<10} {event.thread_name:<20} {str(event.operation):<30} {event.status:<15}"
        )

    print(f"{'-'*80}")
    if events:
        final_time = max(e.end_time for e in events if e.end_time is not None)
        print(f"Total execution time: {final_time:.1f}s")

        # Calculate and display efficiency statistics
        if num_cores:
            # Calculate total work done (sum of all operation durations)
            total_work = sum(
                e.end_time - e.start_time
                for e in events
                if e.status == "completed" and e.end_time is not None
            )

            # Calculate ideal time (if all work was perfectly parallelized)
            ideal_time = total_work / num_cores

            # Calculate actual parallelism achieved
            max_possible_time = final_time * num_cores
            parallelism = total_work / final_time if final_time > 0 else 0
            efficiency = (parallelism / num_cores) * 100 if num_cores > 0 else 0

            print(f"\nEfficiency Statistics:")
            print(f"  Total work: {total_work:.1f}s")
            print(f"  Ideal time (perfect parallelism): {ideal_time:.1f}s")
            print(f"  Average parallelism: {parallelism:.2f} cores")
            print(f"  Efficiency: {efficiency:.1f}% (on {num_cores} cores)")

    print(f"{'='*80}\n")


def print_memory_state(
    memory,
    variables: List[Tuple[str, Optional[any]]] = None,
    title: str = "Final memory state",
):
    """
    Print memory state in a formatted way.

    Args:
        memory: Memory object from simulator
        variables: Optional list of (name, expected_value) tuples to display
                  If None, prints all memory contents
        title: Header title for the output
    """
    print(f"\n{title}:")

    if variables:
        # Print specific variables with optional expected values
        for var_info in variables:
            if isinstance(var_info, tuple):
                name, expected = var_info
                value = memory.read(name)
                if expected is not None:
                    print(f"  {name} = {value} (expected: {expected})")
                else:
                    print(f"  {name} = {value}")
            else:
                # Just a name string
                name = var_info
                value = memory.read(name)
                print(f"  {name} = {value}")
    else:
        # Print all memory contents
        if not memory.data:
            print("  (empty)")
        else:
            for address, value in sorted(memory.data.items()):
                available = "✓" if memory.is_available(address) else "✗"
                print(f"  {address} = {value} [{available}]")
