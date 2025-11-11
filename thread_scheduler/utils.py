#!/usr/bin/env python3
"""
Utility functions for the thread scheduler simulator.
"""

from typing import Dict, List, Tuple, Optional
import inspect


def get_caller_location() -> Tuple[Optional[str], Optional[int]]:
    """
    Get the source file and line number of the user code that called a Device method.

    Walks up the call stack to find the frame that called into the Device class,
    which represents the actual user code creating operations.

    Returns:
        Tuple of (source_file, source_line) or (None, None) if unavailable
    """
    frame = inspect.currentframe()
    if not frame:
        return None, None

    import os

    # Walk up the stack to find a public Device method, then return its caller
    current = frame.f_back

    while current:
        # Check if this frame is a public Device method (not starting with _)
        if "self" in current.f_locals:
            self_obj = current.f_locals["self"]
            if (
                self_obj.__class__.__name__ == "Device"
                and not current.f_code.co_name.startswith("_")
            ):
                # Found a public Device method - return its caller's location
                if current.f_back:
                    return current.f_back.f_code.co_filename, current.f_back.f_lineno

        current = current.f_back

    # Fallback: find first frame outside the package
    package_dir = os.path.dirname(os.path.abspath(__file__))
    current = frame.f_back
    while current:
        filename = current.f_code.co_filename
        if not filename.startswith(package_dir) and not filename.startswith("<"):
            return filename, current.f_lineno
        current = current.f_back

    return None, None


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
    import os

    width = 120
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}")
    print(
        f"{'Time':<12} {'Thread':<20} {'Operation':<40} {'Status':<12} {'Source':<30}"
    )
    print(f"{'-'*width}")

    for event in events:
        time_str = f"{event.start_time:.1f}s"
        if event.status == "completed":
            time_str += f" - {event.end_time:.1f}s"

        # Format source location if available
        source_info = ""
        if event.operation.source_file and event.operation.source_line:
            filename = os.path.basename(event.operation.source_file)
            source_info = f"{filename}:{event.operation.source_line}"

        print(
            f"{time_str:<12} {event.thread_name:<20} {str(event.operation):<40} {event.status:<12} {source_info:<30}"
        )

    print(f"{'-'*width}")
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

    print(f"{'='*width}\n")


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
