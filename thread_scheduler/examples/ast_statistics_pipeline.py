#!/usr/bin/env python3
"""
AST Statistics Pipeline Example

Natural Python syntax version of statistics_pipeline.py.
Parallel computation with 10 threads on 4 cores.
Demonstrates deep dependency chains and core utilization.
"""

import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thread_scheduler import create_simulation, generate_perfetto_trace
from thread_scheduler.utils import (
    print_example_header,
    print_timeline,
)


sim = create_simulation(num_cores=4)
dev = sim.dev


@sim.thread(name="host_init")
def host():
    """Initialize input values."""
    a = 10
    b = 20
    c = 30
    d = 40
    return a, b, c, d


@sim.thread(name="worker1_sum_ab")
def worker1():
    """Compute sum of a and b."""
    sum_ab = a + b
    return sum_ab


@sim.thread(name="worker2_sum_cd")
def worker2():
    """Compute sum of c and d."""
    sum_cd = c + d
    return sum_cd


@sim.thread(name="worker3_prod_ab")
def worker3():
    """Compute product of a and b."""
    prod_ab = a * b
    return prod_ab


@sim.thread(name="worker4_prod_cd")
def worker4():
    """Compute product of c and d."""
    prod_cd = c * d
    return prod_cd


@sim.thread(name="worker5_diff_sums")
def worker5():
    """Compute difference of sums."""
    diff_sums = sum_ab - sum_cd
    return diff_sums


@sim.thread(name="worker6_diff_prods")
def worker6():
    """Compute difference of products."""
    diff_prods = prod_ab - prod_cd
    return diff_prods


@sim.thread(name="worker7_mean")
def worker7():
    """Compute total sum (mean proxy)."""
    total_sum = sum_ab + sum_cd
    return total_sum


@sim.thread(name="worker8_variance")
def worker8():
    """Compute variance proxy."""
    variance = diff_sums * diff_prods
    return variance


@sim.thread(name="worker9_final")
def worker9():
    """Compute final result."""
    final_result = total_sum + variance
    return final_result


if __name__ == "__main__":
    print_example_header(
        title="AST Statistics Pipeline",
        description="Natural Python syntax version with 10 threads.",
        scheduler_info="4 cores",
    )

    events = sim.run()
    print_timeline(events, "AST Statistics Pipeline", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_ast_statistics_pipeline.json")

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
