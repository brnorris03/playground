#!/usr/bin/env python3
"""
Run all examples in sequence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess

examples = [
    "deadlock_demo.py",
    "device_api_demo.py",
    "math_pipeline.py",
    "multi_consumer.py",
    "parallel_math.py",
    "producer_consumer.py",
    "statistics_pipeline.py",
]

print("\n" + "#" * 80)
print("#" + " " * 78 + "#")
print("#" + "THREAD SCHEDULER SIMULATOR - ALL EXAMPLES".center(78) + "#")
print("#" + " " * 78 + "#")
print("#" * 80 + "\n")

for example in examples:
    print(f"\nRunning {example}...")
    print("-" * 80)
    subprocess.run([sys.executable, example], cwd=Path(__file__).parent)
    print()

print("\n" + "#" * 80)
print("All examples completed!")
print("Trace files generated in traces/ directory")
print("View at: https://ui.perfetto.dev")
print("#" * 80 + "\n")
