#!/usr/bin/env python3
"""
Example programs for the thread scheduler simulator.
Demonstrates various patterns including read/write, producer-consumer, and math pipelines.
"""

from simulator import Memory, Thread, Scheduler, create_operation
from perfetto_trace import generate_perfetto_trace
from utils import print_example_header, print_timeline


def example1_simple_read_write():
    """
    Example 1: Simple read/write between host and one worker thread.

    Host writes data, worker waits for it and reads it.
    """
    print_example_header(
        example_num=1,
        title="Simple Read/Write",
        description="Host thread writes data, worker thread waits and reads it.",
        scheduler_info="2 cores",
    )

    memory = Memory()
    scheduler = Scheduler(memory, num_cores=2)

    # Host thread: writes data to address "x"
    host = Thread(
        thread_id=0,
        name="host",
        operations=[
            create_operation("write", "x", 42),
        ],
    )

    # Worker thread: waits for data at "x"
    worker = Thread(
        thread_id=1,
        name="worker",
        operations=[
            create_operation("wait", "x"),
        ],
    )

    scheduler.add_thread(host)
    scheduler.add_thread(worker)

    events = scheduler.run()
    print_timeline(events, "Example 1: Simple Read/Write")
    generate_perfetto_trace(events, "trace_example1.json")

    print(f"Final memory state: x = {memory.read('x')}")


def example2_producer_consumer():
    """
    Example 2: Producer-consumer pattern with multiple threads.

    Producer writes data, multiple consumers wait for it.
    """
    print_example_header(
        example_num=2,
        title="Producer-Consumer Pattern",
        description="One producer writes data, two consumers wait and process it.",
        scheduler_info="3 cores (all threads can run in parallel)",
    )

    memory = Memory()
    scheduler = Scheduler(memory, num_cores=3)

    # Producer thread
    producer = Thread(
        thread_id=0,
        name="producer",
        operations=[
            create_operation("write", "data", 100),
        ],
    )

    # Consumer 1: waits for data, then writes result
    consumer1 = Thread(
        thread_id=1,
        name="consumer1",
        operations=[
            create_operation("wait", "data"),
            create_operation("write", "result1", 200),
        ],
    )

    # Consumer 2: waits for data, then writes result
    consumer2 = Thread(
        thread_id=2,
        name="consumer2",
        operations=[
            create_operation("wait", "data"),
            create_operation("write", "result2", 300),
        ],
    )

    scheduler.add_thread(producer)
    scheduler.add_thread(consumer1)
    scheduler.add_thread(consumer2)

    events = scheduler.run()
    print_timeline(events, "Example 2: Producer-Consumer")
    generate_perfetto_trace(events, "trace_example2.json")

    print(f"Final memory state:")
    print(f"  data = {memory.read('data')}")
    print(f"  result1 = {memory.read('result1')}")
    print(f"  result2 = {memory.read('result2')}")


def example3_math_pipeline():
    """
    Example 3: Math pipeline with dependencies.

    Thread1 writes initial values, Thread2 adds them, Thread3 multiplies result.
    """
    description = (
        "Pipeline of math operations with dependencies.\n"
        "  Thread1: writes a=5, b=3\n"
        "  Thread2: waits for a,b then computes sum=a+b\n"
        "  Thread3: waits for sum, writes c=2, then computes product=sum*c"
    )
    print_example_header(
        example_num=3,
        title="Math Pipeline",
        description=description,
        scheduler_info="3 cores",
    )

    memory = Memory()
    scheduler = Scheduler(memory, num_cores=3)

    # Thread 1: Initialize values
    thread1 = Thread(
        thread_id=0,
        name="initializer",
        operations=[
            create_operation("write", "a", 5),
            create_operation("write", "b", 3),
        ],
    )

    # Thread 2: Add a + b
    thread2 = Thread(
        thread_id=1,
        name="adder",
        operations=[
            create_operation("wait", "a"),
            create_operation("wait", "b"),
            create_operation("add", "a", "b", "sum"),
        ],
    )

    # Thread 3: Multiply sum * c
    thread3 = Thread(
        thread_id=2,
        name="multiplier",
        operations=[
            create_operation("wait", "sum"),
            create_operation("write", "c", 2),
            create_operation("multiply", "sum", "c", "product"),
        ],
    )

    scheduler.add_thread(thread1)
    scheduler.add_thread(thread2)
    scheduler.add_thread(thread3)

    events = scheduler.run()
    print_timeline(events, "Example 3: Math Pipeline")
    generate_perfetto_trace(events, "trace_example3.json")

    print(f"Final memory state:")
    print(f"  a = {memory.read('a')}")
    print(f"  b = {memory.read('b')}")
    print(f"  sum = {memory.read('sum')} (expected: 8)")
    print(f"  c = {memory.read('c')}")
    print(f"  product = {memory.read('product')} (expected: 16)")


def example4_complex_dependencies():
    """
    Example 4: Complex multi-consumer scenario with multiple dependencies.

    Demonstrates a more complex dependency graph with multiple operations.
    """
    description = (
        "Multiple threads with complex dependencies.\n"
        "  Host: writes x=10, y=20\n"
        "  Worker1: computes diff=x-y\n"
        "  Worker2: computes sum=x+y\n"
        "  Worker3: computes result=sum*diff"
    )
    print_example_header(
        example_num=4,
        title="Complex Dependencies",
        description=description,
        scheduler_info="4 cores (maximum parallelism)",
    )

    memory = Memory()
    scheduler = Scheduler(memory, num_cores=4)

    # Host thread: Initialize values
    host = Thread(
        thread_id=0,
        name="host",
        operations=[
            create_operation("write", "x", 10),
            create_operation("write", "y", 20),
        ],
    )

    # Worker 1: Compute difference
    worker1 = Thread(
        thread_id=1,
        name="worker1_subtract",
        operations=[
            create_operation("wait", "x"),
            create_operation("wait", "y"),
            create_operation("subtract", "x", "y", "diff"),
        ],
    )

    # Worker 2: Compute sum
    worker2 = Thread(
        thread_id=2,
        name="worker2_add",
        operations=[
            create_operation("wait", "x"),
            create_operation("wait", "y"),
            create_operation("add", "x", "y", "sum"),
        ],
    )

    # Worker 3: Multiply results
    worker3 = Thread(
        thread_id=3,
        name="worker3_multiply",
        operations=[
            create_operation("wait", "sum"),
            create_operation("wait", "diff"),
            create_operation("multiply", "sum", "diff", "result"),
        ],
    )

    scheduler.add_thread(host)
    scheduler.add_thread(worker1)
    scheduler.add_thread(worker2)
    scheduler.add_thread(worker3)

    events = scheduler.run()
    print_timeline(events, "Example 4: Complex Dependencies")
    generate_perfetto_trace(events, "trace_example4.json")

    print(f"Final memory state:")
    print(f"  x = {memory.read('x')}")
    print(f"  y = {memory.read('y')}")
    print(f"  diff = {memory.read('diff')} (expected: -10)")
    print(f"  sum = {memory.read('sum')} (expected: 30)")
    print(f"  result = {memory.read('result')} (expected: -300)")


def run_all_examples():
    """Run all example scenarios."""
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "THREAD SCHEDULER SIMULATOR - EXAMPLES".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    example1_simple_read_write()
    example2_producer_consumer()
    example3_math_pipeline()
    example4_complex_dependencies()

    print("\n" + "#" * 80)
    print("All examples completed!")
    print("Trace files generated in traces/ directory:")
    print("  - traces/trace_example1.json")
    print("  - traces/trace_example2.json")
    print("  - traces/trace_example3.json")
    print("  - traces/trace_example4.json")
    print("\nView traces at: https://ui.perfetto.dev")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    run_all_examples()
