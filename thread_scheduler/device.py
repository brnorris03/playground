#!/usr/bin/env python3
"""
Device class providing a clean API for creating operations.
"""

from .simulator import Operation, OpType
from .utils import get_caller_location
from .config import OPERATION_DURATIONS
from typing import Any


class Device:
    """
    Device class providing methods for each operation type.

    Usage:
        dev = Device()
        operations = [
            dev.write("x", 10),
            dev.write("y", 20),
            dev.wait("x"),
            dev.add("x", "y", "sum"),
            dev.push("sum"),
        ]
    """

    def __init__(self, default_duration: float = 1.0):
        """
        Initialize device with default operation duration.

        Args:
            default_duration: Default duration for all operations in seconds
        """
        self.default_duration = default_duration

    def _create_op(self, op_type: str, *args, duration: float = None) -> Operation:
        """Helper method to create operations with source location tracking."""
        # Capture caller's source location (automatically finds first frame outside package)
        source_file, source_line = get_caller_location()

        op_type_enum = OpType(op_type)
        return Operation(
            op_type=op_type_enum,
            args=args,
            duration=duration or self.default_duration,
            source_file=source_file,
            source_line=source_line,
        )

    # Synchronization operations

    def wait(self, address: str, duration: float = None) -> Operation:
        """
        Wait for data to be available at address.

        Args:
            address: Memory address to wait for
            duration: Optional custom duration (defaults to config value)

        Returns:
            Wait operation
        """
        return self._create_op(
            "wait",
            address,
            duration=duration if duration is not None else OPERATION_DURATIONS["wait"],
        )

    def push(self, address: str, duration: float = None) -> Operation:
        """
        Make computed data available at address.

        Args:
            address: Memory address to push
            duration: Optional custom duration (defaults to config value)

        Returns:
            Push operation
        """
        return self._create_op(
            "push",
            address,
            duration=duration if duration is not None else OPERATION_DURATIONS["push"],
        )

    # Memory operations

    def write(self, address: str, value: Any, duration: float = None) -> Operation:
        """
        Write value to address and mark as available.

        Args:
            address: Memory address to write to
            value: Value to write
            duration: Optional custom duration (defaults to config value)

        Returns:
            Write operation
        """
        return self._create_op(
            "write",
            address,
            value,
            duration=duration if duration is not None else OPERATION_DURATIONS["write"],
        )

    # Math operations

    def add(
        self, addr1: str, addr2: str, dest: str, duration: float = None
    ) -> Operation:
        """
        Add values from two addresses and store in destination.

        Args:
            addr1: First source address
            addr2: Second source address
            dest: Destination address
            duration: Optional custom duration (defaults to config value)

        Returns:
            Add operation
        """
        return self._create_op(
            "add",
            addr1,
            addr2,
            dest,
            duration=duration if duration is not None else OPERATION_DURATIONS["add"],
        )

    def subtract(
        self, addr1: str, addr2: str, dest: str, duration: float = None
    ) -> Operation:
        """
        Subtract addr2 from addr1 and store in destination.

        Args:
            addr1: First source address (minuend)
            addr2: Second source address (subtrahend)
            dest: Destination address
            duration: Optional custom duration (defaults to device default)

        Returns:
            Subtract operation
        """
        return self._create_op(
            "subtract",
            addr1,
            addr2,
            dest,
            duration=(
                duration if duration is not None else OPERATION_DURATIONS["subtract"]
            ),
        )

    def multiply(
        self, addr1: str, addr2: str, dest: str, duration: float = None
    ) -> Operation:
        """
        Multiply values from two addresses and store in destination.

        Args:
            addr1: First source address
            addr2: Second source address
            dest: Destination address
            duration: Optional custom duration (defaults to config value)

        Returns:
            Multiply operation
        """
        return self._create_op(
            "multiply",
            addr1,
            addr2,
            dest,
            duration=(
                duration if duration is not None else OPERATION_DURATIONS["multiply"]
            ),
        )
