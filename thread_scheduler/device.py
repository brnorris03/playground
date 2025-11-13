#!/usr/bin/env python3
"""
Device class providing a clean API for creating operations.
"""

from .simulator import (
    Operation,
    WaitOp,
    WriteOp,
    PushOp,
    AddOp,
    SubtractOp,
    MultiplyOp,
)
from .utils import get_caller_location, SourceLocation
from .config import OPERATION_DURATIONS
from typing import Any, Optional


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

    def _create_op(
        self,
        op_class: type,
        *args,
        duration: float = None,
        location: Optional[SourceLocation] = None,
        **kwargs
    ) -> Operation:
        """Helper method to create operations with source location tracking."""
        # If source location not provided, capture caller's location
        if location is None:
            location = get_caller_location()

        return op_class(
            *args,
            duration=duration or self.default_duration,
            source_file=location.file,
            source_line=location.line,
            **kwargs,
        )

    # Synchronization operations

    def wait(
        self,
        address: str,
        duration: float = None,
        location: Optional[SourceLocation] = None,
    ) -> Operation:
        """
        Wait for data to be available at address.

        Args:
            address: Memory address to wait for
            duration: Optional custom duration (defaults to config value)
            location: Optional source location for tracking

        Returns:
            Wait operation
        """
        return self._create_op(
            WaitOp,
            address,
            duration=duration if duration is not None else OPERATION_DURATIONS["wait"],
            location=location,
        )

    def push(
        self,
        address: str,
        duration: float = None,
        location: Optional[SourceLocation] = None,
    ) -> Operation:
        """
        Make computed data available at address.

        Args:
            address: Memory address to push
            duration: Optional custom duration (defaults to config value)
            location: Optional source location for tracking


        Returns:
            Push operation
        """
        return self._create_op(
            PushOp,
            address,
            duration=duration if duration is not None else OPERATION_DURATIONS["push"],
            location=location,
        )

    # Memory operations

    def write(
        self,
        address: str,
        value: Any,
        duration: float = None,
        location: Optional[SourceLocation] = None,
    ) -> Operation:
        """
        Write value to address (does NOT mark as available - use push() for that).

        Args:
            address: Memory address to write to
            value: Value to write
            duration: Optional custom duration (defaults to config value)
            location: Optional source location for tracking


        Returns:
            Write operation
        """
        return self._create_op(
            WriteOp,
            address,
            value,
            duration=duration if duration is not None else OPERATION_DURATIONS["write"],
            location=location,
        )

    # Math operations

    def add(
        self,
        addr1: str,
        addr2: str,
        dest: str,
        duration: float = None,
        location: Optional[SourceLocation] = None,
    ) -> Operation:
        """
        Add values from two addresses and store in destination.

        Args:
            addr1: First source address
            addr2: Second source address
            dest: Destination address
            duration: Optional custom duration (defaults to config value)
            location: Optional source location for tracking


        Returns:
            Add operation
        """
        return self._create_op(
            AddOp,
            addr1,
            addr2,
            dest,
            duration=duration if duration is not None else OPERATION_DURATIONS["add"],
            location=location,
        )

    def subtract(
        self,
        addr1: str,
        addr2: str,
        dest: str,
        duration: float = None,
        location: Optional[SourceLocation] = None,
    ) -> Operation:
        """
        Subtract addr2 from addr1 and store in destination.

        Args:
            addr1: First source address (minuend)
            addr2: Second source address (subtrahend)
            dest: Destination address
            duration: Optional custom duration (defaults to device default)
            location: Optional source location for tracking


        Returns:
            Subtract operation
        """
        return self._create_op(
            SubtractOp,
            addr1,
            addr2,
            dest,
            duration=(
                duration if duration is not None else OPERATION_DURATIONS["subtract"]
            ),
            location=location,
        )

    def multiply(
        self,
        addr1: str,
        addr2: str,
        dest: str,
        duration: float = None,
        location: Optional[SourceLocation] = None,
    ) -> Operation:
        """
        Multiply values from two addresses and store in destination.

        Args:
            addr1: First source address
            addr2: Second source address
            dest: Destination address
            duration: Optional custom duration (defaults to config value)
            location: Optional source location for tracking


        Returns:
            Multiply operation
        """
        return self._create_op(
            MultiplyOp,
            addr1,
            addr2,
            dest,
            duration=(
                duration if duration is not None else OPERATION_DURATIONS["multiply"]
            ),
            location=location,
        )
