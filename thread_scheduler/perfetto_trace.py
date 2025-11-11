#!/usr/bin/env python3
"""
Perfetto trace generation for the thread scheduler simulator.
Generates JSON traces compatible with https://ui.perfetto.dev
"""

import json
import os
from typing import List, Dict, Any
from simulator import ExecutionEvent


class PerfettoTraceGenerator:
    """Generates Perfetto-compatible JSON traces from execution events."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add_complete_event(
        self,
        name: str,
        category: str,
        timestamp_us: float,
        duration_us: float,
        pid: int,
        tid: int,
        args: Dict[str, Any] = None,
    ):
        """
        Add a complete event (type 'X') to the trace.

        Args:
            name: Event name
            category: Event category
            timestamp_us: Start timestamp in microseconds
            duration_us: Duration in microseconds
            pid: Process ID
            tid: Thread ID
            args: Optional arguments/metadata
        """
        event = {
            "name": name,
            "cat": category,
            "ph": "X",  # Complete event
            "ts": timestamp_us,
            "dur": duration_us,
            "pid": pid,
            "tid": tid,
        }
        if args:
            event["args"] = args
        self.events.append(event)

    def add_instant_event(
        self,
        name: str,
        category: str,
        timestamp_us: float,
        pid: int,
        tid: int,
        scope: str = "t",
        args: Dict[str, Any] = None,
    ):
        """
        Add an instant event (type 'i') to the trace.

        Args:
            name: Event name
            category: Event category
            timestamp_us: Timestamp in microseconds
            pid: Process ID
            tid: Thread ID
            scope: Scope ('g' for global, 'p' for process, 't' for thread)
            args: Optional arguments/metadata
        """
        event = {
            "name": name,
            "cat": category,
            "ph": "i",  # Instant event
            "ts": timestamp_us,
            "pid": pid,
            "tid": tid,
            "s": scope,
        }
        if args:
            event["args"] = args
        self.events.append(event)

    def add_metadata_event(self, name: str, pid: int, tid: int, args: Dict[str, Any]):
        """Add metadata event for naming threads/processes."""
        event = {
            "name": name,
            "ph": "M",  # Metadata event
            "pid": pid,
            "tid": tid,
            "args": args,
        }
        self.events.append(event)

    def generate_trace_from_events(
        self, execution_events: List[ExecutionEvent], time_scale: float = 1_000_000.0
    ) -> str:
        """
        Generate a Perfetto trace from execution events.

        Args:
            execution_events: List of execution events from the simulator
            time_scale: Scale factor to convert logical time to microseconds
                       (default: 1 second = 1,000,000 microseconds)

        Returns:
            JSON string of the trace
        """
        self.events.clear()

        # Use a single process ID for all threads
        pid = 1

        # Track unique threads and add metadata
        thread_ids = set()
        thread_names = {}
        for event in execution_events:
            if event.thread_id not in thread_ids:
                thread_ids.add(event.thread_id)
                thread_names[event.thread_id] = event.thread_name
                # Add thread name metadata
                self.add_metadata_event(
                    name="thread_name",
                    pid=pid,
                    tid=event.thread_id,
                    args={"name": event.thread_name},
                )

        # Add process name metadata
        self.add_metadata_event(
            name="process_name",
            pid=pid,
            tid=0,
            args={"name": "Thread Scheduler Simulator"},
        )

        # Convert execution events to trace events
        for event in execution_events:
            timestamp_us = event.start_time * time_scale

            if event.status == "completed":
                duration_us = event.duration() * time_scale

                # Create operation description
                op_str = str(event.operation)

                # Add complete event
                self.add_complete_event(
                    name=event.operation.op_type.value,
                    category="operation",
                    timestamp_us=timestamp_us,
                    duration_us=duration_us,
                    pid=pid,
                    tid=event.thread_id,
                    args={
                        "operation": op_str,
                        "thread": event.thread_name,
                        "args": str(event.operation.args),
                    },
                )

            elif event.status == "blocked":
                # Add instant event for blocked operation
                self.add_instant_event(
                    name=f"blocked_{event.operation.op_type.value}",
                    category="blocked",
                    timestamp_us=timestamp_us,
                    pid=pid,
                    tid=event.thread_id,
                    scope="t",
                    args={
                        "operation": str(event.operation),
                        "thread": event.thread_name,
                        "reason": "waiting for data",
                    },
                )

        # Create the trace format
        trace = {
            "traceEvents": self.events,
            "displayTimeUnit": "ms",
            "metadata": {
                "description": "Thread Scheduler Simulator Trace",
                "time_scale": time_scale,
            },
        }

        return json.dumps(trace, indent=2)

    def save_trace(
        self,
        execution_events: List[ExecutionEvent],
        filename: str = "trace.json",
        time_scale: float = 1_000_000.0,
    ):
        """
        Generate and save a Perfetto trace to a file.

        Args:
            execution_events: List of execution events from the simulator
            filename: Output filename (relative to traces/ directory by default)
            time_scale: Scale factor to convert logical time to microseconds
        """
        # Create traces directory if it doesn't exist
        traces_dir = "traces"
        os.makedirs(traces_dir, exist_ok=True)

        # If filename doesn't contain a directory, put it in traces/
        if os.path.dirname(filename) == "":
            filepath = os.path.join(traces_dir, filename)
        else:
            filepath = filename

        trace_json = self.generate_trace_from_events(execution_events, time_scale)
        with open(filepath, "w") as f:
            f.write(trace_json)
        print(f"Trace saved to {filepath}")
        print(f"View at: https://ui.perfetto.dev")


def generate_perfetto_trace(
    execution_events: List[ExecutionEvent],
    filename: str = "trace.json",
    time_scale: float = 1_000_000.0,
):
    """
    Convenience function to generate a Perfetto trace.

    Args:
        execution_events: List of execution events from the simulator
        filename: Output filename (relative to traces/ directory by default)
        time_scale: Scale factor to convert logical time to microseconds
    """
    generator = PerfettoTraceGenerator()
    generator.save_trace(execution_events, filename, time_scale)
