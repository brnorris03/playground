#!/usr/bin/env python3
"""
Decorator-based API for creating simulated threads.
Simplifies thread creation by using Python decorators.
"""

from typing import Callable, List, Optional
from .simulator import Thread, Scheduler, Memory
from .device import Device


class SimulationBuilder:
    """Builder for creating simulations with decorator-based thread definitions."""

    def __init__(self, num_cores: int = 4):
        self.num_cores = num_cores
        self.memory = Memory()
        self.scheduler = Scheduler(self.memory, num_cores=num_cores)
        self.threads = []
        self._thread_counter = 0
        self.dev = Device()  # Device instance for creating operations

    def thread(self, name: Optional[str] = None):
        """
        Decorator to mark a function as a simulated thread.

        The decorated function should return a list of operations.

        Example:
            @sim.thread(name="worker1")
            def worker1():
                return [
                    sim.dev.wait("x"),
                    sim.dev.add("x", "y", "result"),
                    sim.dev.push("result"),
                ]
        """

        def decorator(func: Callable) -> Callable:
            thread_name = name or func.__name__
            thread_id = self._thread_counter
            self._thread_counter += 1

            # Execute the function to get operations
            operations = func()

            # Create and register the thread
            thread = Thread(
                thread_id=thread_id,
                name=thread_name,
                operations=operations,
            )
            self.threads.append(thread)
            self.scheduler.add_thread(thread)

            # Return the original function (though it won't be called again)
            return func

        return decorator

    def run(self):
        """Run the simulation and return events."""
        return self.scheduler.run()

    def get_memory(self):
        """Get the memory object."""
        return self.memory

    def get_scheduler(self):
        """Get the scheduler object."""
        return self.scheduler

    def get_num_cores(self):
        """Get the number of cores."""
        return self.num_cores


def create_simulation(num_cores: int = 4) -> SimulationBuilder:
    """
    Create a new simulation builder.

    Args:
        num_cores: Number of cores for the scheduler

    Returns:
        SimulationBuilder instance

    Example:
        sim = create_simulation(num_cores=4)
        dev = sim.dev

        @sim.thread(name="host")
        def host_thread():
            return [
                dev.write("x", 10),
                dev.write("y", 20),
            ]

        @sim.thread(name="worker")
        def worker_thread():
            return [
                dev.wait("x"),
                dev.wait("y"),
                dev.add("x", "y", "sum"),
                dev.push("sum"),
            ]

        events = sim.run()
    """
    return SimulationBuilder(num_cores=num_cores)
