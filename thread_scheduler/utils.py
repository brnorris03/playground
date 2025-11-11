#!/usr/bin/env python3
"""
Utility functions for the thread scheduler simulator.
"""


def print_example_header(example_num, title, description, scheduler_info):
    """Print a standardized header for examples."""
    print("\n" + "=" * 80)
    print(f"EXAMPLE {example_num}: {title}")
    print("=" * 80)
    print(f"Description: {description}")
    print(f"Scheduler: {scheduler_info}")
    print()


def print_timeline(events, title="Execution Timeline"):
    """Print a console timeline summary."""
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
    print(f"{'='*80}\n")
