#!/usr/bin/env python3
"""
Configuration for the thread scheduler simulator.
"""

# Default operation durations (in seconds)
OPERATION_DURATIONS = {
    "wait": 0,  # Overhead when wait completes
    "write": 1.0,  # Write data to memory
    "push": 0.2,  # Mark data as available
    "add": 1.0,  # Addition operation
    "subtract": 1.0,  # Subtraction operation
    "multiply": 1.0,  # Multiplication operation
}
