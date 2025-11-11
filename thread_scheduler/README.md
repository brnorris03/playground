# Thread Scheduler Simulator

A Python-based multithreaded multicore device simulator that runs in a single Python thread. The simulator models multiple execution threads with various operations, tracks logical timestamps, and generates Perfetto-compatible traces for visualization.

## Features

- **Simulated Operations**: wait, write, add, multiply, subtract
- **Round-robin Scheduling**: Fair scheduling among ready threads
- **Memory Model**: Address-based memory with data availability tracking
- **Logical Timestamps**: Track execution time without real-time delays
- **Perfetto Integration**: Generate traces viewable at [ui.perfetto.dev](https://ui.perfetto.dev)
- **Console Output**: Text-based timeline summary

## Architecture

### Core Components

1. **Memory**: Simulates memory with address-based storage and availability tracking
2. **Thread**: Represents a simulated thread with a sequence of operations
3. **Scheduler**: Round-robin scheduler that manages thread execution
4. **Operations**: 
   - `wait(address)`: Block until data is available (can be preempted)
   - `write(address, value)`: Write data and mark as available (1s duration)
   - `add/multiply/subtract(addr1, addr2, dest)`: Math operations (1s each)

### Scheduling Behavior

- The scheduler uses round-robin scheduling among ready threads
- Operations run to completion except for `wait`, which blocks if data is unavailable
- Blocked threads automatically become ready when their data becomes available
- Each operation has a predetermined duration (default: 1 second)

## Installation

No external dependencies required! The simulator uses only Python standard library.

```bash
cd thread_scheduler
python3 examples.py
```

## Usage

### Running Examples

Run all provided examples:

```bash
python3 examples.py
```

This will:
1. Execute 4 example scenarios
2. Print console timeline summaries
3. Generate trace files in `traces/` directory: `traces/trace_example1.json`, `traces/trace_example2.json`, etc.

### Viewing Traces

1. Open [https://ui.perfetto.dev](https://ui.perfetto.dev) in your browser
2. Click "Open trace file" 
3. Select one of the generated trace files from the `traces/` directory
4. Explore the timeline visualization showing thread execution

### Creating Custom Simulations

```python
from simulator import Memory, Thread, Scheduler, create_operation
from perfetto_trace import generate_perfetto_trace

# Create memory and scheduler
memory = Memory()
scheduler = Scheduler(memory)

# Define threads with operations
host = Thread(
    thread_id=0,
    name="host",
    operations=[
        create_operation("write", "x", 42),
        create_operation("write", "y", 10),
    ]
)

worker = Thread(
    thread_id=1,
    name="worker",
    operations=[
        create_operation("wait", "x"),
        create_operation("wait", "y"),
        create_operation("add", "x", "y", "result"),
    ]
)

# Add threads to scheduler
scheduler.add_thread(host)
scheduler.add_thread(worker)

# Run simulation
events = scheduler.run()

# Generate trace (will be saved to traces/my_trace.json)
generate_perfetto_trace(events, "my_trace.json")

# Print results
for event in events:
    print(f"{event.start_time}s: {event.thread_name} - {event.operation}")
```

## Examples

### Example 1: Simple Read/Write

Host thread writes data, worker thread waits and reads it.

```
Time       Thread               Operation                      Status         
--------------------------------------------------------------------------------
0.0s - 1.0s host                 write('x', 42)                 completed      
0.0s       worker               wait('x')                      blocked        
1.0s - 1.0s worker               wait('x')                      completed      
```

### Example 2: Producer-Consumer

One producer writes data, two consumers wait and process it.

### Example 3: Math Pipeline

Pipeline of math operations with dependencies:
- Thread1: writes a=5, b=3
- Thread2: computes sum=a+b
- Thread3: computes product=sum*c

### Example 4: Complex Dependencies

Multiple threads with complex dependency graphs demonstrating parallel execution and synchronization.

## Operation Reference

### wait(address)
Waits for data to be available at the specified memory address.
- **Blocks** if data is not available
- **Completes immediately** if data is available
- Can be preempted (no time passes when blocked)

### write(address, value)
Writes a value to memory and marks it as available.
- **Duration**: 1 second
- **Side effect**: Makes data available for waiting threads
- Runs to completion (cannot be preempted)

### add(addr1, addr2, dest)
Adds values from two addresses and writes result to destination.
- **Duration**: 1 second
- **Formula**: `dest = memory[addr1] + memory[addr2]`
- Runs to completion

### multiply(addr1, addr2, dest)
Multiplies values from two addresses and writes result to destination.
- **Duration**: 1 second
- **Formula**: `dest = memory[addr1] * memory[addr2]`
- Runs to completion

### subtract(addr1, addr2, dest)
Subtracts second value from first and writes result to destination.
- **Duration**: 1 second
- **Formula**: `dest = memory[addr1] - memory[addr2]`
- Runs to completion

## Perfetto Trace Format

The simulator generates traces in the Chrome Trace Event Format, compatible with Perfetto:

- **Complete Events (X)**: Show operation duration on timeline
- **Instant Events (i)**: Mark blocked operations
- **Metadata Events (M)**: Name threads and processes
- **Time Scale**: Logical seconds converted to microseconds for visualization

Each thread appears as a separate track in the Perfetto UI, making it easy to visualize parallel execution and dependencies.

## Implementation Details

### Round-Robin Scheduling

The scheduler maintains a round-robin index and selects the next ready thread in circular order. This ensures fair scheduling and prevents starvation.

### Memory Availability

Memory addresses have two states:
- **Available**: Data has been written and can be read
- **Unavailable**: No data written yet, `wait` operations will block

### Logical Time

The simulator uses logical time (not real-time):
- Each operation has a predetermined duration
- Time advances only when operations complete
- No actual delays or sleeps occur during simulation

### Deadlock Detection

The simulator detects deadlocks when all threads are either:
- Completed, or
- Blocked waiting for data that will never arrive

## Files

- `simulator.py`: Core simulator implementation
- `perfetto_trace.py`: Perfetto trace generation
- `examples.py`: Example scenarios and demonstrations
- `README.md`: This file

## Further Reading

- [Perfetto Documentation](https://perfetto.dev/docs/)
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)

## Improvements

- Instead of generating JSON, use the Perfetto SDK
