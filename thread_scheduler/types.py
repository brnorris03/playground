#!/usr/bin/env python3
"""
Type definitions for the thread scheduler simulator.
"""

from enum import Enum


class ThreadState(str, Enum):
    """Thread execution states."""

    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class EventStatus(str, Enum):
    """Execution event statuses."""

    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    UNBLOCKED = "unblocked"
