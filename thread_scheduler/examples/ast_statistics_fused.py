#!/usr/bin/env python3
"""
AST Statistics Fused Example

Parallel computation with 5 threads on 2 cores.
Demonstrates operation fusion to reduce thread count.
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


sim = create_simulation(num_cores=2)
dev = sim.dev


@sim.thread(name="init")
def init():
    """Initialize input values."""
    a = 10
    b = 20
    c = 30
    d = 40
    return a, b, c, d


@sim.thread(name="compute_sums")
def compute_sums():
    """Compute sums of pairs."""
    sum_ab = a + b
    sum_cd = c + d
    return sum_ab, sum_cd


@sim.thread(name="compute_products")
def compute_products():
    """Compute products of pairs."""
    prod_ab = a * b
    prod_cd = c * d
    return prod_ab, prod_cd


@sim.thread(name="compute_diffs")
def compute_diffs():
    """Compute differences."""
    diff_sums = sum_ab - sum_cd
    diff_prods = prod_ab - prod_cd
    return diff_sums, diff_prods


@sim.thread(name="compute_final")
def compute_final():
    """Compute final result."""
    total = sum_ab + sum_cd
    variance = diff_sums * diff_prods
    result = total + variance
    return result


if __name__ == "__main__":
    print_example_header(
        title="AST Statistics Fused",
        description="Natural Python syntax with 5 threads.",
        scheduler_info="2 cores",
    )

    events = sim.run()
    print_timeline(events, "AST Statistics Fused", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_ast_statistics_fused.json")

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
    print(f"\nFinal result:")
    print(f"  result = {memory.read('result')} (expected: 40100)")
    print("=" * 80 + "\n")
