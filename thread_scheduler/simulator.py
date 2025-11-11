#!/usr/bin/env python3
"""
Core simulator for multithreaded multicore device simulation.
Runs in a single Python thread with logical timestamps.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class OpType(Enum):
    """Operation types supported by the simulator."""

    WAIT = "wait"
    WRITE = "write"
    ADD = "add"
    MULTIPLY = "multiply"
    SUBTRACT = "subtract"


@dataclass
class Operation:
    """Represents a single operation to be executed."""

    op_type: OpType
    args: Tuple[Any, ...]
    duration: float = 1.0  # Default 1 second for all ops

    def __repr__(self):
        return f"{self.op_type.value}({', '.join(map(str, self.args))})"


@dataclass
class ExecutionEvent:
    """Records an execution event for tracing."""

    thread_id: int
    thread_name: str
    operation: Operation
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, blocked

    def duration(self) -> float:
        if self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0


class Memory:
    """Simulates memory with address-based storage and availability tracking."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.available: Dict[str, bool] = {}

    def write(self, address: str, value: Any):
        """Write data to an address and mark it as available."""
        self.data[address] = value
        self.available[address] = True

    def read(self, address: str) -> Any:
        """Read data from an address."""
        return self.data.get(address)

    def is_available(self, address: str) -> bool:
        """Check if data is available at an address."""
        return self.available.get(address, False)

    def reset(self):
        """Reset memory state."""
        self.data.clear()
        self.available.clear()


class Thread:
    """Represents a simulated thread with a sequence of operations."""

    def __init__(self, thread_id: int, name: str, operations: List[Operation]):
        self.thread_id = thread_id
        self.name = name
        self.operations = operations
        self.pc = 0  # Program counter
        self.state = "ready"  # ready, running, blocked, completed
        self.blocked_on: Optional[str] = None  # Address we're waiting for

    def current_operation(self) -> Optional[Operation]:
        """Get the current operation to execute."""
        if self.pc < len(self.operations):
            return self.operations[self.pc]
        return None

    def advance(self):
        """Move to the next operation."""
        self.pc += 1
        if self.pc >= len(self.operations):
            self.state = "completed"

    def is_ready(self, memory: Memory) -> bool:
        """Check if this thread is ready to execute."""
        if self.state == "completed":
            return False

        if self.state == "blocked":
            # Check if the blocking condition is resolved
            if self.blocked_on and memory.is_available(self.blocked_on):
                self.state = "ready"
                self.blocked_on = None
                return True
            return False

        return self.state == "ready"

    def reset(self):
        """Reset thread to initial state."""
        self.pc = 0
        self.state = "ready"
        self.blocked_on = None


class Scheduler:
    """Round-robin scheduler for simulated threads with multicore support."""

    def __init__(self, memory: Memory, num_cores: int = 4):
        self.memory = memory
        self.threads: List[Thread] = []
        self.current_time = 0.0
        self.events: List[ExecutionEvent] = []
        self.last_scheduled_idx = -1  # For round-robin
        self.num_cores = num_cores
        self.running_operations: List[Tuple[Thread, Operation, float]] = (
            []
        )  # (thread, op, end_time)

    def add_thread(self, thread: Thread):
        """Add a thread to the scheduler."""
        self.threads.append(thread)

    def get_ready_threads(self) -> List[Tuple[int, Thread]]:
        """Get all threads that are ready to run."""
        ready = []
        for idx, thread in enumerate(self.threads):
            if thread.is_ready(self.memory):
                ready.append((idx, thread))
        return ready

    def select_next_thread(self) -> Optional[Tuple[int, Thread]]:
        """Select the next thread to run using round-robin scheduling."""
        ready_threads = self.get_ready_threads()
        if not ready_threads:
            return None

        # Round-robin: find the next thread after last_scheduled_idx
        # that is ready to run
        num_threads = len(self.threads)
        for _ in range(num_threads):
            self.last_scheduled_idx = (self.last_scheduled_idx + 1) % num_threads
            for idx, thread in ready_threads:
                if idx == self.last_scheduled_idx:
                    return (idx, thread)

        # Fallback: return first ready thread
        return ready_threads[0]

    def execute_operation(self, thread: Thread, operation: Operation) -> bool:
        """
        Execute a single operation.
        Returns True if operation completed, False if blocked.
        """
        if operation.op_type == OpType.WAIT:
            address = operation.args[0]
            if self.memory.is_available(address):
                # Data is available, operation completes
                return True
            else:
                # Block the thread
                thread.state = "blocked"
                thread.blocked_on = address
                return False

        elif operation.op_type == OpType.WRITE:
            address, value = operation.args
            self.memory.write(address, value)
            return True

        elif operation.op_type == OpType.ADD:
            addr1, addr2, dest = operation.args
            val1 = self.memory.read(addr1)
            val2 = self.memory.read(addr2)
            result = val1 + val2
            self.memory.write(dest, result)
            return True

        elif operation.op_type == OpType.MULTIPLY:
            addr1, addr2, dest = operation.args
            val1 = self.memory.read(addr1)
            val2 = self.memory.read(addr2)
            result = val1 * val2
            self.memory.write(dest, result)
            return True

        elif operation.op_type == OpType.SUBTRACT:
            addr1, addr2, dest = operation.args
            val1 = self.memory.read(addr1)
            val2 = self.memory.read(addr2)
            result = val1 - val2
            self.memory.write(dest, result)
            return True

        return False

    def step(self) -> bool:
        """
        Execute one scheduling step with multicore support.
        Returns True if any thread made progress, False if all are blocked/completed.
        """
        # First, complete any operations that finished at current time
        newly_completed = []
        still_running = []

        for thread, operation, end_time in self.running_operations:
            if end_time <= self.current_time:
                # Operation completed
                newly_completed.append((thread, operation, end_time))
            else:
                still_running.append((thread, operation, end_time))

        self.running_operations = still_running

        # Mark completed operations and advance threads
        for thread, operation, end_time in newly_completed:
            # Execute the operation (perform side effects like memory writes)
            self.execute_operation(thread, operation)

            # Find and update the event
            for event in self.events:
                if (
                    event.thread_id == thread.thread_id
                    and event.operation == operation
                    and event.status == "running"
                ):
                    event.end_time = end_time
                    event.status = "completed"
                    break

            thread.advance()
            thread.state = "ready"

        # Check all blocked threads to see if they can now proceed
        for thread in self.threads:
            if thread.state == "blocked":
                thread.is_ready(self.memory)  # This will update state if unblocked

        # Try to schedule new operations on available cores
        available_cores = self.num_cores - len(self.running_operations)
        scheduled_any = False

        while available_cores > 0:
            # Get ready threads that aren't currently running
            running_thread_ids = {t.thread_id for t, _, _ in self.running_operations}
            ready_threads = [
                (idx, t)
                for idx, t in self.get_ready_threads()
                if t.thread_id not in running_thread_ids
            ]

            if not ready_threads:
                break

            # Select next thread using round-robin
            selected = None
            num_threads = len(self.threads)
            for _ in range(num_threads):
                self.last_scheduled_idx = (self.last_scheduled_idx + 1) % num_threads
                for idx, thread in ready_threads:
                    if idx == self.last_scheduled_idx:
                        selected = (idx, thread)
                        break
                if selected:
                    break

            if not selected:
                selected = ready_threads[0]

            idx, thread = selected
            operation = thread.current_operation()
            if operation is None:
                thread.state = "completed"
                break

            # Try to execute the operation
            start_time = self.current_time
            event = ExecutionEvent(
                thread_id=thread.thread_id,
                thread_name=thread.name,
                operation=operation,
                start_time=start_time,
                status="running",
            )

            # Check if operation can start
            if operation.op_type == OpType.WAIT:
                address = operation.args[0]
                if self.memory.is_available(address):
                    # Data available, wait completes immediately
                    event.end_time = start_time
                    event.status = "completed"
                    thread.advance()
                    thread.state = (
                        "ready"  # Make sure thread is ready for next operation
                    )
                    self.events.append(event)
                    scheduled_any = True
                    # Don't break - continue to try scheduling more threads
                else:
                    # Block the thread
                    thread.state = "blocked"
                    thread.blocked_on = address
                    event.status = "blocked"
                    event.end_time = start_time
                    self.events.append(event)
                    scheduled_any = True
                    # Don't break - continue to try scheduling other threads
            else:
                # Non-wait operation: schedule it to run
                end_time = start_time + operation.duration
                self.running_operations.append((thread, operation, end_time))
                thread.state = "running"
                self.events.append(event)
                available_cores -= 1
                scheduled_any = True

        # Advance time to next event
        if self.running_operations:
            next_completion = min(
                end_time for _, _, end_time in self.running_operations
            )
            self.current_time = next_completion
            return True
        elif scheduled_any:
            return True
        else:
            return False

    def run(self, max_steps: int = 10000) -> List[ExecutionEvent]:
        """
        Run the scheduler until all threads complete or max_steps reached.
        Returns the list of execution events.
        """
        steps = 0
        while steps < max_steps:
            made_progress = self.step()
            if not made_progress:
                # Check if all threads are completed
                all_completed = all(t.state == "completed" for t in self.threads)
                if all_completed:
                    break
                # All threads are blocked, deadlock
                print(f"Warning: Deadlock detected at time {self.current_time}")
                break
            steps += 1

        if steps >= max_steps:
            print(f"Warning: Reached max steps ({max_steps})")

        return self.events

    def reset(self):
        """Reset the scheduler state."""
        self.current_time = 0.0
        self.events.clear()
        self.last_scheduled_idx = -1
        self.running_operations.clear()
        self.memory.reset()
        for thread in self.threads:
            thread.reset()


def create_operation(op_type: str, *args, duration: float = 1.0) -> Operation:
    """Helper function to create operations.

    Args:
        op_type: Type of operation (wait, write, add, multiply, subtract)
        *args: Variable arguments depending on operation type
        duration: Duration of the operation in seconds (default: 1.0)
    """
    op_type_enum = OpType(op_type)
    return Operation(op_type=op_type_enum, args=args, duration=duration)
