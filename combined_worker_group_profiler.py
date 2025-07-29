#!/usr/bin/env python3
"""
Combined Worker Group Profiler

This profiler combines both:
1. Worker Group Startup Profiling (using callbacks)
2. Ray Core Operations Profiling (get/wait operations)

This gives you comprehensive insights into both the high-level startup phases
and the low-level Ray operations that might be causing bottlenecks.
"""

import time
import logging
import traceback
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import defaultdict

import ray
from ray.train.v2._internal.execution.callback import WorkerGroupCallback
from ray.train.v2._internal.execution.worker_group import WorkerGroup

# Global registry to avoid closure serialization issues
_global_combined_profiler = None

@dataclass
class RayOperation:
    """Represents a Ray operation (get/wait) with timing details."""

    operation_type: str  # "get" or "wait"
    start_time: float
    end_time: float = None
    duration: float = None
    object_refs: List[str] = None
    num_objects: int = 0
    timeout: Optional[float] = None
    fetch_local: bool = True
    stack_trace: str = None
    worker_id: str = None
    node_ip: str = None
    error: str = None
    phase_context: str = None  # Which worker group phase this operation occurred in

    def finish(self, error: str = None):
        """Mark the operation as finished and calculate duration."""
        self.end_time = time.monotonic()
        self.duration = self.end_time - self.start_time
        if error:
            self.error = error


@dataclass
class StartupPhase:
    """Represents a phase in the worker group startup process."""

    name: str
    start_time: float
    end_time: float = None
    duration: float = None
    details: Dict[str, Any] = None
    error: str = None
    ray_operations: List[RayOperation] = None

    def finish(self, details: Dict[str, Any] = None, error: str = None):
        """Mark the phase as finished and calculate duration."""
        self.end_time = time.monotonic()
        self.duration = self.end_time - self.start_time
        if details:
            self.details = details
        if error:
            self.error = error


class CombinedWorkerGroupProfiler(WorkerGroupCallback):
    """
    Combined profiler that tracks both worker group startup phases and Ray core operations.
    """

    def __init__(self, output_dir: str = "/tmp/combined_worker_group_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Track phases and operations
        self.phases: List[StartupPhase] = []
        self.current_phase: Optional[StartupPhase] = None
        self.ray_operations: List[RayOperation] = []
        self.current_operations: Dict[int, RayOperation] = {}
        self.operation_counter = 0

        # Statistics
        self.stats = {
            "total_startup_time": 0.0,
            "phase_breakdown": {},
            "slow_phases": [],
            "failed_phases": [],
            "total_get_operations": 0,
            "total_wait_operations": 0,
            "total_get_time": 0.0,
            "total_wait_time": 0.0,
            "slow_operations": [],
            "failed_operations": [],
        }

        # Track operation patterns
        self.operation_patterns = defaultdict(int)

        # Track worker group state
        self.worker_group_info = {}

        # Instrument Ray core functions
        self._instrument_ray_core()

        self.logger.info("CombinedWorkerGroupProfiler initialized")

    def _setup_logging(self):
        """Setup detailed logging for the profiler."""
        logger = logging.getLogger("CombinedWorkerGroupProfiler")

        return logger

    def _get_stack_trace(self, max_depth: int = 10) -> str:
        """Get a simplified stack trace for the current operation."""
        try:
            import traceback

            stack = traceback.extract_stack()
            # Filter out profiler-related frames
            relevant_frames = []
            for frame in stack[-max_depth:]:
                if "combined_worker_group_profiler" not in frame.filename:
                    relevant_frames.append(
                        f"{frame.filename}:{frame.lineno} in {frame.name}"
                    )
            return " -> ".join(relevant_frames[-5:])  # Last 5 relevant frames
        except Exception:
            return "Stack trace unavailable"

    def _get_worker_info(self) -> Tuple[str, str]:
        """Get current worker ID and node IP."""
        try:
            if ray.is_initialized():
                runtime_context = ray.get_runtime_context()
                worker_id = str(runtime_context.get_worker_id())
                node_ip = runtime_context.get_node_ip_address()
                return worker_id, node_ip
        except Exception:
            pass
        return "unknown", "unknown"

    def _start_phase(self, name: str, details: Dict[str, Any] = None) -> StartupPhase:
        """Start tracking a new phase."""
        phase = StartupPhase(
            name=name,
            start_time=time.monotonic(),
            details=details or {},
            ray_operations=[],
        )

        self.current_phase = phase
        self.phases.append(phase)

        self.logger.info(f"Started phase: {name}")
        if details:
            self.logger.debug(f"Phase details: {details}")

        return phase

    def _end_phase(self, name: str, details: Dict[str, Any] = None, error: str = None):
        """End tracking the current phase."""
        if self.current_phase and self.current_phase.name == name:
            self.current_phase.finish(details, error)

            # Update statistics
            self.stats["phase_breakdown"][name] = self.current_phase.duration

            # Track slow phases
            if self.current_phase.duration > 5.0:
                self.stats["slow_phases"].append(self.current_phase)

            # Track failed phases
            if error:
                self.stats["failed_phases"].append(self.current_phase)

            # Log phase completion
            self.logger.info(
                f"Completed phase: {name} in {self.current_phase.duration:.3f}s"
            )
            if error:
                self.logger.error(f"Phase {name} failed: {error}")

            self.current_phase = None

    def _start_ray_operation(
        self,
        operation_type: str,
        object_refs: List[ray.ObjectRef],
        timeout: Optional[float] = None,
        fetch_local: bool = True,
    ) -> int:
        """Start tracking a Ray operation."""
        operation_id = self.operation_counter
        self.operation_counter += 1

        worker_id, node_ip = self._get_worker_info()

        # Determine which phase this operation belongs to
        phase_context = self.current_phase.name if self.current_phase else "unknown"

        operation = RayOperation(
            operation_type=operation_type,
            start_time=time.monotonic(),
            object_refs=[str(ref) for ref in object_refs] if object_refs else [],
            num_objects=len(object_refs) if object_refs else 0,
            timeout=timeout,
            fetch_local=fetch_local,
            stack_trace=self._get_stack_trace(),
            worker_id=worker_id,
            node_ip=node_ip,
            phase_context=phase_context,
        )

        self.current_operations[operation_id] = operation
        self.ray_operations.append(operation)

        # Add to current phase if available
        if self.current_phase:
            self.current_phase.ray_operations.append(operation)

        # Log operation start
        self.logger.debug(
            f"Started {operation_type} operation {operation_id} in phase '{phase_context}' with "
            f"{len(object_refs)} objects, timeout={timeout}, fetch_local={fetch_local}"
        )

        return operation_id

    def _end_ray_operation(self, operation_id: int, error: str = None):
        """End tracking a Ray operation."""
        if operation_id in self.current_operations:
            operation = self.current_operations[operation_id]
            operation.finish(error)

            # Update statistics
            if operation.operation_type == "get":
                self.stats["total_get_operations"] += 1
                self.stats["total_get_time"] += operation.duration
            elif operation.operation_type == "wait":
                self.stats["total_wait_operations"] += 1
                self.stats["total_wait_time"] += operation.duration

            # Track slow operations
            if operation.duration > 1.0:
                self.stats["slow_operations"].append(operation)

            # Track failed operations
            if error:
                self.stats["failed_operations"].append(operation)

            # Track operation patterns
            pattern_key = f"{operation.operation_type}_{operation.num_objects}_{operation.timeout}"
            self.operation_patterns[pattern_key] += 1

            # Log operation end
            self.logger.info(
                f"Completed {operation.operation_type} operation {operation_id} in "
                f"{operation.duration:.3f}s with {operation.num_objects} objects "
                f"(phase: {operation.phase_context})"
            )

            del self.current_operations[operation_id]

    def _instrument_ray_core(self):
        """Instrument Ray's core get and wait functions."""
        import ray._private.worker as ray_worker
        import functools

        # Store original functions
        self.original_get = ray_worker.get
        self.original_wait = ray_worker.wait

        # Set global profiler reference
        global _global_combined_profiler
        _global_combined_profiler = self

        # Instrument ray.get
        @functools.wraps(self.original_get)
        def instrumented_get(object_refs, *, timeout=None):
            # Convert to list if single object ref
            if not isinstance(object_refs, list):
                object_refs = [object_refs]

            # Use global profiler reference to avoid closure
            profiler = _global_combined_profiler
            if profiler is None:
                return self.original_get(object_refs, timeout=timeout)

            operation_id = profiler._start_ray_operation("get", object_refs, timeout)

            try:
                result = self.original_get(object_refs, timeout=timeout)
                profiler._end_ray_operation(operation_id)
                return result
            except Exception as e:
                profiler._end_ray_operation(operation_id, str(e))
                raise

        # Instrument ray.wait
        @functools.wraps(self.original_wait)
        def instrumented_wait(
            ray_waitables, *, num_returns=1, timeout=None, fetch_local=True
        ):
            # Use global profiler reference to avoid closure
            profiler = _global_combined_profiler
            if profiler is None:
                return self.original_wait(
                    ray_waitables,
                    num_returns=num_returns,
                    timeout=timeout,
                    fetch_local=fetch_local,
                )

            operation_id = profiler._start_ray_operation(
                "wait", ray_waitables, timeout, fetch_local
            )

            try:
                result = self.original_wait(
                    ray_waitables,
                    num_returns=num_returns,
                    timeout=timeout,
                    fetch_local=fetch_local,
                )
                profiler._end_ray_operation(operation_id)
                return result
            except Exception as e:
                profiler._end_ray_operation(operation_id, str(e))
                raise

        # Apply the instrumented versions
        ray_worker.get = instrumented_get
        ray_worker.wait = instrumented_wait

        self.logger.info("Instrumented Ray core get and wait functions")

    def _restore_ray_core(self):
        """Restore original Ray core functions."""
        import ray._private.worker as ray_worker

        if hasattr(self, "original_get"):
            ray_worker.get = self.original_get
        if hasattr(self, "original_wait"):
            ray_worker.wait = self.original_wait

        # Clear global profiler reference
        global _global_combined_profiler
        _global_combined_profiler = None

        self.logger.info("Restored original Ray core functions")

    def _get_worker_group_info(self, worker_group: WorkerGroup) -> Dict[str, Any]:
        """Extract information about the worker group."""
        try:
            info = {
                "num_workers": len(worker_group.workers)
                if hasattr(worker_group, "workers")
                else 0,
                "worker_group_id": str(worker_group.worker_group_id)
                if hasattr(worker_group, "worker_group_id")
                else "unknown",
                "placement_group_id": str(worker_group.placement_group.id)
                if hasattr(worker_group, "placement_group")
                and worker_group.placement_group
                else "unknown",
            }

            # Try to get more details if available
            if hasattr(worker_group, "workers") and worker_group.workers:
                worker_info = []
                for i, worker in enumerate(worker_group.workers):
                    worker_detail = {
                        "index": i,
                        "actor_id": str(worker._actor_id)
                        if hasattr(worker, "_actor_id")
                        else "unknown",
                        "node_id": str(worker._node_id)
                        if hasattr(worker, "_node_id")
                        else "unknown",
                    }
                    worker_info.append(worker_detail)
                info["worker_details"] = worker_info

            return info
        except Exception as e:
            self.logger.warning(f"Could not extract worker group info: {e}")
            return {"error": str(e)}

    # WorkerGroupCallback methods
    def after_worker_group_start(self, worker_group: WorkerGroup):
        """Called after the worker group has started."""
        self.logger.info("Worker group start callback triggered")

        # Get worker group information
        worker_info = self._get_worker_group_info(worker_group)
        self.worker_group_info = worker_info

        # End the startup phase if it's still running
        if self.current_phase and self.current_phase.name == "worker_group_startup":
            self._end_phase(
                "worker_group_startup",
                {"worker_group_info": worker_info, "callback_triggered": True},
            )

        # Start a new phase for post-startup operations
        self._start_phase("post_startup_operations", {"worker_group_info": worker_info})

    def after_worker_group_training_start(self, worker_group: WorkerGroup):
        """Called after training has started on the worker group."""
        self.logger.info("Worker group training start callback triggered")

        # End the post-startup phase
        if self.current_phase and self.current_phase.name == "post_startup_operations":
            self._end_phase("post_startup_operations", {"training_started": True})

        # Start training phase
        self._start_phase("training_execution", {"training_started": True})

    def before_worker_group_shutdown(self, worker_group: WorkerGroup):
        """Called before the worker group shuts down."""
        self.logger.info("Worker group shutdown callback triggered")

        # End any current phase
        if self.current_phase:
            self._end_phase(self.current_phase.name, {"shutdown_initiated": True})

        # Start shutdown phase
        self._start_phase("worker_group_shutdown", {"shutdown_initiated": True})

    def after_worker_group_shutdown(self, worker_group: WorkerGroup):
        """Called after the worker group has shut down."""
        self.logger.info("Worker group shutdown completed")

        # End shutdown phase
        if self.current_phase and self.current_phase.name == "worker_group_shutdown":
            self._end_phase("worker_group_shutdown", {"shutdown_completed": True})

        # Calculate total startup time
        if self.phases:
            first_phase = self.phases[0]
            last_phase = self.phases[-1]
            if first_phase.start_time and last_phase.end_time:
                self.stats["total_startup_time"] = (
                    last_phase.end_time - first_phase.start_time
                )

        # Save results
        self.save_results()

    def after_worker_group_poll_status(self, worker_group: WorkerGroup):
        """Called after polling the status of the worker group."""
        # This is called frequently, so we'll just log it at debug level
        self.logger.debug("Worker group poll status callback triggered")

    def save_results(self):
        """Save detailed profiling results."""
        # Save phases to JSON
        phases_data = [asdict(phase) for phase in self.phases]
        with open(self.output_dir / "startup_phases.json", "w") as f:
            json.dump(phases_data, f, indent=2)

        # Save Ray operations to JSON
        operations_data = [asdict(op) for op in self.ray_operations]
        with open(self.output_dir / "ray_operations.json", "w") as f:
            json.dump(operations_data, f, indent=2)

        # Save worker group info
        with open(self.output_dir / "worker_group_info.json", "w") as f:
            json.dump(self.worker_group_info, f, indent=2)

        # Generate reports
        self._generate_summary_report()
        self._generate_bottleneck_analysis()
        self._generate_detailed_timing_report()
        self._generate_ray_operations_analysis()

        self.logger.info(f"Combined profiling results saved to {self.output_dir}")

    def _generate_summary_report(self):
        """Generate a comprehensive summary report."""
        with open(self.output_dir / "combined_summary.txt", "w") as f:
            f.write("Combined Worker Group Profiling Summary\n")
            f.write("=" * 50 + "\n\n")

            # Worker group startup summary
            f.write("WORKER GROUP STARTUP:\n")
            f.write(f"  Total startup time: {self.stats['total_startup_time']:.3f}s\n")
            f.write(f"  Total phases tracked: {len(self.phases)}\n")
            f.write(f"  Slow phases (>5s): {len(self.stats['slow_phases'])}\n")
            f.write(f"  Failed phases: {len(self.stats['failed_phases'])}\n\n")

            f.write("  Phase breakdown:\n")
            for phase_name, duration in self.stats["phase_breakdown"].items():
                f.write(f"    {phase_name}: {duration:.3f}s\n")
            f.write("\n")

            # Ray operations summary
            f.write("RAY CORE OPERATIONS:\n")
            f.write(f"  Total get operations: {self.stats['total_get_operations']}\n")
            f.write(f"  Total wait operations: {self.stats['total_wait_operations']}\n")

            if self.stats["total_get_operations"] > 0:
                avg_get_time = (
                    self.stats["total_get_time"] / self.stats["total_get_operations"]
                )
                f.write(f"  Average get time: {avg_get_time:.3f}s\n")

            if self.stats["total_wait_operations"] > 0:
                avg_wait_time = (
                    self.stats["total_wait_time"] / self.stats["total_wait_operations"]
                )
                f.write(f"  Average wait time: {avg_wait_time:.3f}s\n")

            f.write(f"  Slow operations (>1s): {len(self.stats['slow_operations'])}\n")
            f.write(f"  Failed operations: {len(self.stats['failed_operations'])}\n\n")

            # Worker group info
            if self.worker_group_info:
                f.write("WORKER GROUP INFO:\n")
                for key, value in self.worker_group_info.items():
                    if key != "worker_details":
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

    def _generate_bottleneck_analysis(self):
        """Generate comprehensive bottleneck analysis."""
        with open(self.output_dir / "combined_bottleneck_analysis.txt", "w") as f:
            f.write("Combined Bottleneck Analysis\n")
            f.write("=" * 30 + "\n\n")

            # Analyze slow phases
            slow_phases = [
                phase
                for phase in self.phases
                if phase.duration and phase.duration > 5.0
            ]
            if slow_phases:
                f.write(f"SLOW PHASES ({len(slow_phases)} found):\n")
                for phase in slow_phases:
                    f.write(f"  {phase.name}: {phase.duration:.3f}s\n")
                    if phase.ray_operations:
                        f.write(
                            f"    Ray operations in this phase: {len(phase.ray_operations)}\n"
                        )
                        slow_ops_in_phase = [
                            op
                            for op in phase.ray_operations
                            if op.duration and op.duration > 1.0
                        ]
                        if slow_ops_in_phase:
                            f.write(
                                f"    Slow operations in this phase: {len(slow_ops_in_phase)}\n"
                            )
                f.write("\n")

            # Analyze slow Ray operations
            slow_ops = [
                op for op in self.ray_operations if op.duration and op.duration > 1.0
            ]
            if slow_ops:
                f.write(f"SLOW RAY OPERATIONS ({len(slow_ops)} found):\n")

                # Group by phase
                ops_by_phase = defaultdict(list)
                for op in slow_ops:
                    ops_by_phase[op.phase_context].append(op)

                for phase_name, ops in ops_by_phase.items():
                    f.write(f"  Phase '{phase_name}': {len(ops)} slow operations\n")
                    for op in ops[:3]:  # Show top 3 per phase
                        f.write(
                            f"    {op.operation_type}: {op.duration:.3f}s ({op.num_objects} objects)\n"
                        )
                f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            if slow_phases or slow_ops:
                f.write("1. Check Ray cluster health and resource availability\n")
                f.write("2. Monitor object store memory usage and pressure\n")
                f.write("3. Check network connectivity between nodes\n")
                f.write("4. Consider reducing the number of workers\n")
                f.write("5. Check for resource contention on worker nodes\n")
                f.write("6. Monitor Ray dashboard for cluster metrics\n")
                f.write(
                    "7. Consider using placement groups for better resource allocation\n"
                )
                f.write("8. Check for large object transfers causing delays\n")
                f.write("9. Monitor serialization/deserialization overhead\n")
            else:
                f.write("No significant bottlenecks detected. Performance is good.\n")

    def _generate_detailed_timing_report(self):
        """Generate a detailed timing report."""
        with open(self.output_dir / "detailed_timing.txt", "w") as f:
            f.write("Detailed Combined Timing Report\n")
            f.write("=" * 30 + "\n\n")

            for i, phase in enumerate(self.phases):
                f.write(f"Phase {i+1}: {phase.name}\n")
                f.write(f"  Start time: {phase.start_time:.3f}s\n")
                f.write(f"  End time: {phase.end_time:.3f}s\n")
                f.write(f"  Duration: {phase.duration:.3f}s\n")

                if phase.details:
                    f.write(f"  Details: {phase.details}\n")

                if phase.error:
                    f.write(f"  Error: {phase.error}\n")

                if phase.ray_operations:
                    f.write(
                        f"  Ray operations in this phase: {len(phase.ray_operations)}\n"
                    )
                    total_op_time = sum(op.duration or 0 for op in phase.ray_operations)
                    f.write(f"  Total Ray operation time: {total_op_time:.3f}s\n")

                    # Show slow operations in this phase
                    slow_ops = [
                        op
                        for op in phase.ray_operations
                        if op.duration and op.duration > 1.0
                    ]
                    if slow_ops:
                        f.write(f"  Slow operations in this phase: {len(slow_ops)}\n")
                        for op in slow_ops[:3]:  # Show top 3
                            f.write(
                                f"    {op.operation_type}: {op.duration:.3f}s ({op.num_objects} objects)\n"
                            )

                f.write("\n")

    def _generate_ray_operations_analysis(self):
        """Generate detailed Ray operations analysis."""
        with open(self.output_dir / "ray_operations_analysis.txt", "w") as f:
            f.write("Ray Operations Analysis\n")
            f.write("=" * 20 + "\n\n")

            # Operation patterns
            f.write("Most common operation patterns:\n")
            sorted_patterns = sorted(
                self.operation_patterns.items(), key=lambda x: x[1], reverse=True
            )

            for pattern, count in sorted_patterns[:20]:
                f.write(f"  {pattern}: {count} times\n")

            f.write("\nPattern interpretation:\n")
            f.write("- get_1_None: Single object get with no timeout\n")
            f.write("- get_4_None: Get 4 objects with no timeout\n")
            f.write("- wait_2_5.0: Wait for 2 objects with 5s timeout\n")
            f.write("- wait_1_None: Wait for 1 object with no timeout\n\n")

            # Operations by phase
            f.write("Operations by phase:\n")
            ops_by_phase = defaultdict(list)
            for op in self.ray_operations:
                ops_by_phase[op.phase_context].append(op)

            for phase_name, ops in ops_by_phase.items():
                f.write(f"  {phase_name}: {len(ops)} operations\n")
                if ops:
                    total_time = sum(op.duration or 0 for op in ops)
                    avg_time = total_time / len(ops)
                    f.write(
                        f"    Total time: {total_time:.3f}s, Average: {avg_time:.3f}s\n"
                    )

            f.write("\n")


@contextmanager
def profile_worker_group_combined(
    output_dir: str = "/tmp/combined_worker_group_profiling",
):
    """
    Context manager for combined worker group profiling.

    Args:
        output_dir: Directory to save profiling results
    """
    profiler = CombinedWorkerGroupProfiler(output_dir)

    try:
        # Start the initial phase
        profiler._start_phase(
            "worker_group_startup",
            {"profiler_initialized": True, "output_dir": str(output_dir)},
        )

        yield profiler
    finally:
        # End any remaining phase
        if profiler.current_phase:
            profiler._end_phase(
                profiler.current_phase.name, {"profiler_finished": True}
            )

        # Restore Ray core functions
        profiler._restore_ray_core()

        # Save results
        profiler.save_results()


def profile_worker_group_combined_example():
    """
    Example of combined worker group profiling.
    """
    print("Combined Worker Group Profiler")
    print("=" * 40)
    print("This profiler combines:")
    print("1. Worker Group Startup Profiling (using callbacks)")
    print("2. Ray Core Operations Profiling (get/wait operations)")
    print()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Training function
    def training_function():
        import time

        print("Training function started")

        # Create some remote objects that will trigger ray.get/ray.wait
        @ray.remote
        def remote_task():
            time.sleep(0.5)
            return "task completed"

        # This will trigger ray.get operations
        results = ray.get([remote_task.remote() for _ in range(4)])
        print(f"Remote tasks completed: {results}")

        time.sleep(1)
        print("Training function completed")

    # Profile worker group startup using combined approach
    with profile_worker_group_combined(
        "/tmp/combined_worker_group_profiling"
    ) as profiler:
        try:
            from ray.train.v2 import DataParallelTrainer

            trainer = DataParallelTrainer(
                train_loop_per_worker=training_function,
                scaling_config={"num_workers": 2, "use_gpu": False},
                run_config={"name": "combined_worker_group_profiling_test"},
            )

            # Add the profiler as a callback
            trainer.add_callbacks([profiler])

            print("Starting training with combined profiling...")
            result = trainer.fit()
            print("Training completed successfully!")

        except Exception as e:
            print(f"Training failed: {e}")
            traceback.print_exc()

    print("\nCombined profiling results saved to /tmp/combined_worker_group_profiling")
    print("Check the following files for detailed analysis:")
    print("  - startup_phases.json: Worker group phase timing")
    print("  - ray_operations.json: Ray get/wait operation timing")
    print("  - worker_group_info.json: Worker group configuration")
    print("  - combined_summary.txt: Comprehensive summary")
    print("  - combined_bottleneck_analysis.txt: Bottleneck analysis")
    print("  - detailed_timing.txt: Detailed timing breakdown")
    print("  - ray_operations_analysis.txt: Ray operations analysis")
    print("  - combined_profiling.log: Detailed logs")


if __name__ == "__main__":
    profile_worker_group_combined_example()
