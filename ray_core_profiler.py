#!/usr/bin/env python3
"""
Focused profiler for Ray's core get and wait operations.

This profiler specifically targets the bottlenecks identified in:
- worker.py:2743 (ray.get)
- worker.py:2944 (ray.wait)

These are the core Ray operations that are likely causing slow worker group startup.
"""

import time
import functools
import logging
import traceback
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import threading
from collections import defaultdict

import ray
from ray import ObjectRef


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

    def finish(self, error: str = None):
        """Mark the operation as finished and calculate duration."""
        self.end_time = time.monotonic()
        self.duration = self.end_time - self.start_time
        if error:
            self.error = error


class RayCoreProfiler:
    """
    Profiler specifically designed to track Ray's core get and wait operations.
    """

    def __init__(self, output_dir: str = "/tmp/ray_core_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Track operations
        self.operations: List[RayOperation] = []
        self.current_operations: Dict[int, RayOperation] = {}
        self.operation_counter = 0

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_get_operations": 0,
            "total_wait_operations": 0,
            "total_get_time": 0.0,
            "total_wait_time": 0.0,
            "slow_operations": [],  # Operations taking > 1 second
            "failed_operations": [],
        }

        # Track operation patterns
        self.operation_patterns = defaultdict(int)

    def _setup_logging(self):
        """Setup detailed logging for the profiler."""
        logger = logging.getLogger("RayCoreProfiler")
        logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.output_dir / "ray_core_profiling.log")
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _get_stack_trace(self, max_depth: int = 10) -> str:
        """Get a simplified stack trace for the current operation."""
        try:
            import traceback

            stack = traceback.extract_stack()
            # Filter out profiler-related frames
            relevant_frames = []
            for frame in stack[-max_depth:]:
                if "ray_core_profiler" not in frame.filename:
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

    def _start_operation(
        self,
        operation_type: str,
        object_refs: List[ObjectRef],
        timeout: Optional[float] = None,
        fetch_local: bool = True,
    ) -> int:
        """Start tracking a Ray operation."""
        with self.lock:
            operation_id = self.operation_counter
            self.operation_counter += 1

            worker_id, node_ip = self._get_worker_info()

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
            )

            self.current_operations[operation_id] = operation
            self.operations.append(operation)

            # Log operation start
            self.logger.debug(
                f"Started {operation_type} operation {operation_id} with "
                f"{len(object_refs)} objects, timeout={timeout}, fetch_local={fetch_local}"
            )

            return operation_id

    def _end_operation(self, operation_id: int, error: str = None):
        """End tracking a Ray operation."""
        with self.lock:
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
                    f"{operation.duration:.3f}s with {operation.num_objects} objects"
                )

                del self.current_operations[operation_id]

    def instrument_ray_core(self):
        """Instrument Ray's core get and wait functions."""
        import ray._private.worker as ray_worker

        # Store original functions
        self.original_get = ray_worker.get
        self.original_wait = ray_worker.wait

        # Instrument ray.get
        @functools.wraps(self.original_get)
        def instrumented_get(object_refs, *, timeout=None):
            # Convert to list if single object ref
            if not isinstance(object_refs, list):
                object_refs = [object_refs]

            operation_id = self._start_operation("get", object_refs, timeout)

            try:
                result = self.original_get(object_refs, timeout=timeout)
                self._end_operation(operation_id)
                return result
            except Exception as e:
                self._end_operation(operation_id, str(e))
                raise

        # Instrument ray.wait
        @functools.wraps(self.original_wait)
        def instrumented_wait(
            ray_waitables, *, num_returns=1, timeout=None, fetch_local=True
        ):
            operation_id = self._start_operation(
                "wait", ray_waitables, timeout, fetch_local
            )

            try:
                result = self.original_wait(
                    ray_waitables,
                    num_returns=num_returns,
                    timeout=timeout,
                    fetch_local=fetch_local,
                )
                self._end_operation(operation_id)
                return result
            except Exception as e:
                self._end_operation(operation_id, str(e))
                raise

        # Apply the instrumented versions
        ray_worker.get = instrumented_get
        ray_worker.wait = instrumented_wait

        self.logger.info("Instrumented Ray core get and wait functions")

    def restore_ray_core(self):
        """Restore original Ray core functions."""
        import ray._private.worker as ray_worker

        if hasattr(self, "original_get"):
            ray_worker.get = self.original_get
        if hasattr(self, "original_wait"):
            ray_worker.wait = self.original_wait

        self.logger.info("Restored original Ray core functions")

    def save_results(self):
        """Save detailed profiling results."""
        # Save operations to JSON
        operations_data = [asdict(op) for op in self.operations]
        with open(self.output_dir / "ray_operations.json", "w") as f:
            json.dump(operations_data, f, indent=2)

        # Generate summary report
        self._generate_summary_report()

        # Generate bottleneck analysis
        self._generate_bottleneck_analysis()

        # Generate operation patterns report
        self._generate_patterns_report()

        self.logger.info(f"Profiling results saved to {self.output_dir}")

    def _generate_summary_report(self):
        """Generate a summary report of Ray operations."""
        with open(self.output_dir / "ray_operations_summary.txt", "w") as f:
            f.write("Ray Core Operations Profiling Summary\n")
            f.write("=" * 40 + "\n\n")

            total_operations = len(self.operations)
            f.write(f"Total operations tracked: {total_operations}\n")
            f.write(f"Get operations: {self.stats['total_get_operations']}\n")
            f.write(f"Wait operations: {self.stats['total_wait_operations']}\n\n")

            if self.stats["total_get_operations"] > 0:
                avg_get_time = (
                    self.stats["total_get_time"] / self.stats["total_get_operations"]
                )
                f.write(f"Average get time: {avg_get_time:.3f}s\n")

            if self.stats["total_wait_operations"] > 0:
                avg_wait_time = (
                    self.stats["total_wait_time"] / self.stats["total_wait_operations"]
                )
                f.write(f"Average wait time: {avg_wait_time:.3f}s\n")

            f.write(f"Slow operations (>1s): {len(self.stats['slow_operations'])}\n")
            f.write(f"Failed operations: {len(self.stats['failed_operations'])}\n\n")

            # Show top 10 slowest operations
            if self.operations:
                sorted_ops = sorted(
                    self.operations, key=lambda x: x.duration or 0, reverse=True
                )
                f.write("Top 10 slowest operations:\n")
                for i, op in enumerate(sorted_ops[:10]):
                    f.write(
                        f"  {i+1}. {op.operation_type} operation: {op.duration:.3f}s "
                        f"({op.num_objects} objects, timeout={op.timeout})\n"
                    )
                    if op.stack_trace:
                        f.write(f"     Stack: {op.stack_trace}\n")
                f.write("\n")

    def _generate_bottleneck_analysis(self):
        """Generate bottleneck analysis and recommendations."""
        with open(self.output_dir / "ray_bottleneck_analysis.txt", "w") as f:
            f.write("Ray Core Operations Bottleneck Analysis\n")
            f.write("=" * 40 + "\n\n")

            # Analyze slow operations
            slow_ops = [
                op for op in self.operations if op.duration and op.duration > 1.0
            ]
            if slow_ops:
                f.write(f"Found {len(slow_ops)} slow operations (>1s):\n\n")

                # Group by operation type
                get_ops = [op for op in slow_ops if op.operation_type == "get"]
                wait_ops = [op for op in slow_ops if op.operation_type == "wait"]

                if get_ops:
                    f.write(f"Slow GET operations ({len(get_ops)}):\n")
                    avg_get_duration = sum(op.duration for op in get_ops) / len(get_ops)
                    f.write(f"  Average duration: {avg_get_duration:.3f}s\n")
                    f.write("  Common patterns:\n")
                    f.write("    - Large object transfers\n")
                    f.write("    - Network latency between nodes\n")
                    f.write("    - Object store pressure\n")
                    f.write("    - Serialization/deserialization overhead\n\n")

                if wait_ops:
                    f.write(f"Slow WAIT operations ({len(wait_ops)}):\n")
                    avg_wait_duration = sum(op.duration for op in wait_ops) / len(
                        wait_ops
                    )
                    f.write(f"  Average duration: {avg_wait_duration:.3f}s\n")
                    f.write("  Common patterns:\n")
                    f.write("    - Waiting for slow tasks to complete\n")
                    f.write("    - Resource contention\n")
                    f.write("    - Network delays in object fetching\n")
                    f.write("    - Object store eviction pressure\n\n")

                # Recommendations
                f.write("Recommendations:\n")
                f.write("1. Check object store memory usage and pressure\n")
                f.write("2. Monitor network latency between nodes\n")
                f.write("3. Consider reducing object sizes or using object spilling\n")
                f.write("4. Check for resource contention on worker nodes\n")
                f.write("5. Consider using placement groups to co-locate workers\n")
                f.write("6. Monitor Ray dashboard for cluster health metrics\n")
            else:
                f.write(
                    "No slow operations detected. Ray operations are performing well.\n"
                )

    def _generate_patterns_report(self):
        """Generate a report on operation patterns."""
        with open(self.output_dir / "ray_operation_patterns.txt", "w") as f:
            f.write("Ray Operation Patterns Analysis\n")
            f.write("=" * 30 + "\n\n")

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
            f.write("- wait_1_None: Wait for 1 object with no timeout\n")


@contextmanager
def profile_ray_core_operations(output_dir: str = "/tmp/ray_core_profiling"):
    """
    Context manager for profiling Ray core operations.

    Args:
        output_dir: Directory to save profiling results
    """
    profiler = RayCoreProfiler(output_dir)

    try:
        # Instrument Ray core functions
        profiler.instrument_ray_core()
        yield profiler
    finally:
        # Restore original functions
        profiler.restore_ray_core()
        # Save results
        profiler.save_results()


def profile_worker_group_with_ray_core_profiling():
    """
    Example of profiling worker group startup with Ray core operation tracking.
    """
    print("Ray Core Operations Profiler")
    print("=" * 40)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Training function
    def simple_train_fn():
        import time

        time.sleep(1)

    # Profile Ray core operations during worker group startup
    with profile_ray_core_operations(
        "/tmp/ray_core_worker_group_profiling"
    ) as profiler:
        try:
            from ray.train.v2 import DataParallelTrainer

            trainer = DataParallelTrainer(
                train_loop_per_worker=simple_train_fn,
                scaling_config={"num_workers": 2, "use_gpu": False},
                run_config={"name": "ray_core_profiling_test"},
            )

            print("Starting training with Ray core operation profiling...")
            result = trainer.fit()
            print("Training completed successfully!")

        except Exception as e:
            print(f"Training failed: {e}")
            traceback.print_exc()

    print("\nRay core profiling results saved to /tmp/ray_core_worker_group_profiling")
    print("Check the following files for detailed analysis:")
    print("  - ray_operations.json: Detailed operation timing")
    print("  - ray_operations_summary.txt: Summary report")
    print("  - ray_bottleneck_analysis.txt: Bottleneck analysis")
    print("  - ray_operation_patterns.txt: Operation patterns")


if __name__ == "__main__":
    profile_worker_group_with_ray_core_profiling()
