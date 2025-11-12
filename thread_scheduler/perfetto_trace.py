#!/usr/bin/env python3
"""
Perfetto trace generation for the thread scheduler simulator.
Generates JSON traces compatible with https://ui.perfetto.dev
"""

import json
import os
from typing import List, Dict, Any
from .simulator import ExecutionEvent
from .types import EventStatus
from .utils import format_source_location


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
        color: str = None,
        alpha: float = None,
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
            color: Optional color name (e.g., 'grey', 'thread_state_sleeping')
            alpha: Optional alpha/transparency (0.0-1.0, where 0.5 is 50% transparent)
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
        if color:
            event["cname"] = color
        if alpha is not None:
            # Perfetto uses 0-255 for alpha, where 255 is opaque
            event["alpha"] = int(alpha * 255)
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

        # Group events by thread to track blocked periods
        blocked_periods = {}  # thread_id -> (start_time, operation)

        # Convert execution events to trace events
        for event in execution_events:
            timestamp_us = event.start_time * time_scale

            if event.status == EventStatus.COMPLETED:
                duration_us = event.duration() * time_scale

                # Create operation description
                op_str = str(event.operation)

                # Build args dict with source location if available
                args_dict = {
                    "operation": op_str,
                    "thread": event.thread_name,
                }

                # Build event name with source location if available
                event_name = event.operation.op_type.value
                source_loc = format_source_location(
                    event.operation.source_file, event.operation.source_line
                )
                if source_loc:
                    args_dict["source"] = source_loc
                    # Add source to event name for visibility in timeline
                    event_name = f"{event.operation.op_type.value} @ {source_loc}"

                # Add complete event
                self.add_complete_event(
                    name=event_name,
                    category="operation",
                    timestamp_us=timestamp_us,
                    duration_us=duration_us,
                    pid=pid,
                    tid=event.thread_id,
                    args=args_dict,
                )

            elif event.status == EventStatus.BLOCKED:
                # Record the start of a blocked period
                blocked_periods[event.thread_id] = (event.start_time, event)

            elif event.status == EventStatus.UNBLOCKED:
                # End the blocked period and create a wait block
                if event.thread_id in blocked_periods:
                    block_start_time, block_event = blocked_periods[event.thread_id]
                    block_duration = event.start_time - block_start_time

                    # Build args dict
                    wait_args = {
                        "operation": str(event.operation),
                        "thread": event.thread_name,
                        "reason": "waiting for data",
                    }

                    # Add source location if available
                    source_loc = format_source_location(
                        event.operation.source_file, event.operation.source_line
                    )
                    if source_loc:
                        wait_args["source"] = source_loc
                        wait_name = f"wait (blocked) @ {source_loc}"
                    else:
                        wait_name = "wait (blocked)"

                    # Create a duration event for the blocked period (neutral gray, subtle)
                    self.add_complete_event(
                        name=wait_name,
                        category="wait",
                        timestamp_us=block_start_time * time_scale,
                        duration_us=block_duration * time_scale,
                        pid=pid,
                        tid=event.thread_id,
                        args=wait_args,
                        color="generic_work",  # Neutral medium gray (125, 125, 125)
                        alpha=0.3,  # 30% opacity
                    )

                    del blocked_periods[event.thread_id]

        # Flush any remaining blocked periods (for threads that never unblocked)
        for thread_id, (block_start_time, block_event) in blocked_periods.items():
            # Create wait block for threads still blocked at end of simulation
            final_time_for_block = max(
                (e.end_time for e in execution_events if e.end_time is not None),
                default=block_start_time,
            )
            block_duration = final_time_for_block - block_start_time

            # Ensure minimum duration for visibility in Perfetto (e.g., 1 second for deadlocked threads)
            if block_duration == 0:
                block_duration = 0.5  # arbitrary minimum duration

            wait_args = {
                "operation": str(block_event.operation),
                "thread": block_event.thread_name,
                "reason": "waiting for data (never unblocked)",
            }

            source_loc = format_source_location(
                block_event.operation.source_file, block_event.operation.source_line
            )
            if source_loc:
                wait_args["source"] = source_loc
                wait_name = f"wait (blocked) @ {source_loc}"
            else:
                wait_name = "wait (blocked)"

            self.add_complete_event(
                name=wait_name,
                category="wait",
                timestamp_us=block_start_time * time_scale,
                duration_us=block_duration * time_scale,
                pid=pid,
                tid=thread_id,
                args=wait_args,
                color="generic_work",
                alpha=0.3,
            )

        # Add clear END OF TRACE marker for all threads
        final_time = 0.0
        if execution_events:
            final_time = max(
                e.end_time for e in execution_events if e.end_time is not None
            )

        end_marker_time = (final_time + 0.1) * time_scale
        for tid in thread_ids:
            self.add_instant_event(
                name="🏁 END OF TRACE 🏁",
                category="marker",
                timestamp_us=end_marker_time,
                pid=pid,
                tid=tid,
                scope="t",
                args={
                    "message": "Simulation completed",
                    "final_time": f"{final_time}s",
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
            filename: Output filename (relative to traces/ directory in current working dir)
            time_scale: Scale factor to convert logical time to microseconds
        """
        # Create traces directory in current working directory if it doesn't exist
        traces_dir = os.path.join(os.getcwd(), "traces")
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
