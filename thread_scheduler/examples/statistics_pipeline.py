#!/usr/bin/env python3
"""
Statistics Pipeline Example

Advanced parallel computation with 10 threads on 4 cores.
Tests deep dependency chains and maximum core utilization.
"""

import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thread_scheduler import create_simulation, generate_perfetto_trace
from thread_scheduler.utils import print_example_header, print_timeline, print_memory_state

import sys


sim = create_simulation(num_cores=4)
dev = sim.dev  # Get device instance


@sim.thread(name="host_init")
def host():
    return [
        dev.write("a", 10),
        dev.write("b", 20),
        dev.write("c", 30),
        dev.write("d", 40),
    ]


@sim.thread(name="worker1_sum_ab")
def worker1():
    return [
        dev.wait("a"),
        dev.wait("b"),
        dev.add("a", "b", "sum_ab"),
        dev.push("sum_ab"),
    ]


@sim.thread(name="worker2_sum_cd")
def worker2():
    return [
        dev.wait("c"),
        dev.wait("d"),
        dev.add("c", "d", "sum_cd"),
        dev.push("sum_cd"),
    ]


@sim.thread(name="worker3_prod_ab")
def worker3():
    return [
        dev.wait("a"),
        dev.wait("b"),
        dev.multiply("a", "b", "prod_ab"),
        dev.push("prod_ab"),
    ]


@sim.thread(name="worker4_prod_cd")
def worker4():
    return [
        dev.wait("c"),
        dev.wait("d"),
        dev.multiply("c", "d", "prod_cd"),
        dev.push("prod_cd"),
    ]


@sim.thread(name="worker5_diff_sums")
def worker5():
    return [
        dev.wait("sum_ab"),
        dev.wait("sum_cd"),
        dev.subtract("sum_ab", "sum_cd", "diff_sums"),
        dev.push("diff_sums"),
    ]


@sim.thread(name="worker6_diff_prods")
def worker6():
    return [
        dev.wait("prod_ab"),
        dev.wait("prod_cd"),
        dev.subtract("prod_ab", "prod_cd", "diff_prods"),
        dev.push("diff_prods"),
    ]


@sim.thread(name="worker7_mean")
def worker7():
    return [
        dev.wait("sum_ab"),
        dev.wait("sum_cd"),
        dev.add("sum_ab", "sum_cd", "total_sum"),
        dev.push("total_sum"),
    ]


@sim.thread(name="worker8_variance")
def worker8():
    return [
        dev.wait("diff_sums"),
        dev.wait("diff_prods"),
        dev.multiply("diff_sums", "diff_prods", "variance"),
        dev.push("variance"),
    ]


@sim.thread(name="worker9_final")
def worker9():
    return [
        dev.wait("total_sum"),
        dev.wait("variance"),
        dev.add("total_sum", "variance", "final_result"),
        dev.push("final_result"),
    ]


if __name__ == "__main__":
    print_example_header(
        title="Advanced Statistics Pipeline",
        description="Parallel computation pipeline with 10 threads and deep dependencies.",
        scheduler_info="4 cores (maximum parallelism)",
    )

    events = sim.run()
    print_timeline(events, "Statistics Pipeline Example", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_statistics_pipeline.json")

    print("\nComputation Results:")
    print("=" * 80)
    memory = sim.get_memory()
    print(
        f"Input data: a={memory.read('a')}, b={memory.read('b')}, c={memory.read('c')}, d={memory.read('d')}"
    )
    print(f"\nIntermediate results:")
    print(f"  sum_ab = {memory.read('sum_ab')} (expected: 30)")
    print(f"  sum_cd = {memory.read('sum_cd')} (expected: 70)")
    print(f"  prod_ab = {memory.read('prod_ab')} (expected: 200)")
    print(f"  prod_cd = {memory.read('prod_cd')} (expected: 1200)")
    print(f"\nDerived values:")
    print(f"  diff_sums = {memory.read('diff_sums')} (expected: -40)")
    print(f"  diff_prods = {memory.read('diff_prods')} (expected: -1000)")
    print(f"  total_sum = {memory.read('total_sum')} (expected: 100)")
    print(f"  variance = {memory.read('variance')} (expected: 40000)")
    print(f"\nFinal result:")
    print(f"  final_result = {memory.read('final_result')} (expected: 40100)")
    print("=" * 80 + "\n")

    # Parallelism analysis
    completed_events = [
        e for e in events if e.status == "completed" and e.duration() > 0
    ]
    if completed_events:
        total_time = max(e.end_time for e in completed_events)
        total_work = sum(e.duration() for e in completed_events)
        parallelism = total_work / total_time if total_time > 0 else 0

        print(f"Parallelism Analysis:")
        print(f"  Total execution time: {total_time:.1f}s")
        print(f"  Total work (sum of all operations): {total_work:.1f}s")
        print(f"  Average parallelism: {parallelism:.2f} cores utilized")
        print(f"  Efficiency: {(parallelism / 4) * 100:.1f}% (on 4 cores)")
        print()
