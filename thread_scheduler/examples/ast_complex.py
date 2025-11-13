#!/usr/bin/env python3
"""
AST Complex Expression Example

Demonstrates complex nested expressions with natural Python syntax.
Computes: result = (a + b) * (c - d) + (a * c) - (b * d)

Note: Uses unique variable names for each thread's output to avoid conflicts
in global scope. Proper scoping (distinguishing local vs external variables)
would allow reusing variable names like 'x', 'y', 'result' across threads.
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


sim = create_simulation(num_cores=8)
dev = sim.dev


@sim.thread(name="init")
def init():
    """Initialize input values."""
    a = 10
    b = 20
    c = 30
    d = 5
    return a, b, c, d


@sim.thread(name="term1")
def compute_term1():
    """Compute first term."""
    term1 = (a + b) * (c - d)
    return term1


@sim.thread(name="term2")
def compute_term2():
    """Compute second term."""
    term2 = a * c
    return term2


@sim.thread(name="term3")
def compute_term3():
    """Compute third term."""
    term3 = b * d
    return term3


@sim.thread(name="final")
def compute_final():
    """Compute final result."""
    result = term1 + term2 - term3
    return result


if __name__ == "__main__":
    print_example_header(
        title="AST Complex Expression",
        description="Complex nested expression: (a+b) * (c-d) + (a*c) - (b*d)",
        scheduler_info="8 cores",
    )

    events = sim.run()
    print_timeline(events, "AST Complex Expression", num_cores=sim.get_num_cores())
    generate_perfetto_trace(events, "trace_ast_complex.json")

    print("\nComputation Results:")
    print("=" * 80)
    memory = sim.get_memory()
    a = memory.read("a")
    b = memory.read("b")
    c = memory.read("c")
    d = memory.read("d")

    print(f"Inputs: a={a}, b={b}, c={c}, d={d}")
    print(f"\nIntermediate terms:")
    print(
        f"  term1 = (a + b) * (c - d) = ({a} + {b}) * ({c} - {d}) = {memory.read('term1')}"
    )
    print(f"  term2 = a * c = {a} * {c} = {memory.read('term2')}")
    print(f"  term3 = b * d = {b} * {d} = {memory.read('term3')}")

    result = memory.read("result")
    expected = (a + b) * (c - d) + (a * c) - (b * d)
    print(f"\nFinal result:")
    print(f"  result = term1 + term2 - term3 = {result}")
    print(f"  Expected: {expected}")
    print(f"  Match: {'✓' if result == expected else '✗'}")
    print("=" * 80 + "\n")
