#!/usr/bin/env python3
"""
Robust Worker Group Startup Profiler v2

This version uses Ray Train callbacks instead of monkey patching to avoid
override issues and provide better visibility into the startup process.
"""

import os

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import defaultdict

from ray.train.v2._internal.execution.callback import WorkerGroupCallback
from ray.train.v2._internal.execution.worker_group import WorkerGroup

import cProfile
import pstats
import sys


@dataclass
class StartupPhase:
    """Represents a phase in the worker group startup process."""

    name: str
    start_time: float
    end_time: float = None
    duration: float = None
    details: Dict[str, Any] = None
    error: str = None

    def finish(self, details: Dict[str, Any] = None, error: str = None):
        """Mark the phase as finished and calculate duration."""
        self.end_time = time.monotonic()
        self.duration = self.end_time - self.start_time
        if details:
            self.details = details
        if error:
            self.error = error


class WorkerGroupStartupProfilerV2(WorkerGroupCallback):
    """
    Enhanced profiler for worker group startup using Ray Train callbacks.
    This version avoids monkey patching issues by using the callback system.
    """

    def __init__(self, output_dir: str = "/tmp/worker_group_startup_profiling_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Track phases
        self.phases: List[StartupPhase] = []
        self.current_phase: Optional[StartupPhase] = None

        # Statistics
        self.stats = {
            "total_startup_time": 0.0,
            "phase_breakdown": {},
            "slow_phases": [],  # Phases taking > 5 seconds
            "failed_phases": [],
        }

        # Track detailed timing
        self.detailed_timings = defaultdict(list)

        # Track worker group state
        self.worker_group_info = {}

        self.logger.info("WorkerGroupStartupProfilerV2 initialized")

    def _setup_logging(self):
        """Setup detailed logging for the profiler."""
        logger = logging.getLogger("WorkerGroupStartupProfilerV2")
        return logger

    def _start_phase(self, name: str, details: Dict[str, Any] = None) -> StartupPhase:
        """Start tracking a new phase."""
        phase = StartupPhase(
            name=name, start_time=time.monotonic(), details=details or {}
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

    @contextmanager
    def on_worker_group_start(self):
        print(
            ">>> wg v2 callbacks are connected correctly and use to profile the startup time"
        )
        """Measure time taken to start worker group."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()

        yield

        if self.profiler:
            self.profiler.disable()

            # Generate readable stats
            stats = pstats.Stats(self.profiler)
            self.profiler.dump_stats(str(self.output_dir / "cprofile_stats.prof"))
            # works, once get it can use snakemake to generate the stats
            stats_file = self.output_dir / "cprofile_stats.txt"
            with open(stats_file, "w") as f:
                # Redirect stdout to capture stats output
                old_stdout = sys.stdout
                sys.stdout = f
                stats.sort_stats("cumulative")
                stats.print_stats(50)  # Top 50 functions
                stats.print_callers(20)  # Top 20 callers
                sys.stdout = old_stdout

    def after_worker_group_start(self, worker_group: WorkerGroup):
        """Called after the worker group has started."""
        print(
            ">>> checking if the v2 callbacks are connected correctly for Worker group start callback triggered"
        )
        self.logger.info("Worker group after wg start callback triggered")
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

        # Save worker group info
        with open(self.output_dir / "worker_group_info.json", "w") as f:
            json.dump(self.worker_group_info, f, indent=2)

        # Generate summary report
        self._generate_summary_report()

        # Generate bottleneck analysis
        self._generate_bottleneck_analysis()

        # Generate detailed timing report
        self._generate_detailed_timing_report()

        self.logger.info(f"Profiling results saved to {self.output_dir}")

    def _generate_summary_report(self):
        """Generate a summary report of the startup process."""
        with open(self.output_dir / "startup_summary.txt", "w") as f:
            f.write("Worker Group Startup Profiling Summary (v2)\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total startup time: {self.stats['total_startup_time']:.3f}s\n")
            f.write(f"Total phases tracked: {len(self.phases)}\n")
            f.write(f"Slow phases (>5s): {len(self.stats['slow_phases'])}\n")
            f.write(f"Failed phases: {len(self.stats['failed_phases'])}\n\n")

            f.write("Phase breakdown:\n")
            for phase_name, duration in self.stats["phase_breakdown"].items():
                f.write(f"  {phase_name}: {duration:.3f}s\n")
            f.write("\n")

            # Show worker group info
            if self.worker_group_info:
                f.write("Worker group information:\n")
                for key, value in self.worker_group_info.items():
                    if key != "worker_details":
                        f.write(f"  {key}: {value}\n")

                if "worker_details" in self.worker_group_info:
                    f.write("  Worker details:\n")
                    for worker in self.worker_group_info["worker_details"]:
                        f.write(f"    Worker {worker['index']}: {worker}\n")
                f.write("\n")

            # Show top 5 slowest phases
            if self.phases:
                sorted_phases = sorted(
                    self.phases, key=lambda x: x.duration or 0, reverse=True
                )
                f.write("Top 5 slowest phases:\n")
                for i, phase in enumerate(sorted_phases[:5]):
                    f.write(f"  {i+1}. {phase.name}: {phase.duration:.3f}s\n")
                    if phase.error:
                        f.write(f"     Error: {phase.error}\n")
                f.write("\n")

    def _generate_bottleneck_analysis(self):
        """Generate bottleneck analysis and recommendations."""
        with open(self.output_dir / "bottleneck_analysis.txt", "w") as f:
            f.write("Worker Group Startup Bottleneck Analysis (v2)\n")
            f.write("=" * 50 + "\n\n")

            # Analyze slow phases
            slow_phases = [
                phase
                for phase in self.phases
                if phase.duration and phase.duration > 5.0
            ]
            if slow_phases:
                f.write(f"Found {len(slow_phases)} slow phases (>5s):\n\n")

                for phase in slow_phases:
                    f.write(f"Slow phase: {phase.name}\n")
                    f.write(f"  Duration: {phase.duration:.3f}s\n")
                    if phase.error:
                        f.write(f"  Error: {phase.error}\n")
                    if phase.details:
                        f.write(f"  Details: {phase.details}\n")
                    f.write("\n")

                # Recommendations based on slow phases
                f.write("Recommendations:\n")
                f.write("1. Check Ray cluster health and resource availability\n")
                f.write("2. Monitor object store memory usage\n")
                f.write("3. Check network connectivity between nodes\n")
                f.write("4. Consider reducing the number of workers\n")
                f.write("5. Check for resource contention on worker nodes\n")
                f.write("6. Monitor Ray dashboard for cluster metrics\n")
                f.write(
                    "7. Consider using placement groups for better resource allocation\n"
                )
            else:
                f.write(
                    "No slow phases detected. Worker group startup is performing well.\n"
                )

    def _generate_detailed_timing_report(self):
        """Generate a detailed timing report."""
        with open(self.output_dir / "detailed_timing.txt", "w") as f:
            f.write("Detailed Worker Group Startup Timing (v2)\n")
            f.write("=" * 40 + "\n\n")

            for i, phase in enumerate(self.phases):
                f.write(f"Phase {i+1}: {phase.name}\n")
                f.write(f"  Start time: {phase.start_time:.3f}s\n")
                f.write(f"  End time: {phase.end_time:.3f}s\n")
                f.write(f"  Duration: {phase.duration:.3f}s\n")

                if phase.details:
                    f.write(f"  Details: {phase.details}\n")

                if phase.error:
                    f.write(f"  Error: {phase.error}\n")

                f.write("\n")


@contextmanager
def profile_worker_group_startup_v2(
    output_dir: str = "/tmp/worker_group_startup_profiling_v2",
):
    """
    Context manager for profiling worker group startup using callbacks.

    Args:
        output_dir: Directory to save profiling results
    """
    profiler = WorkerGroupStartupProfilerV2(output_dir)

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

        # Save results
        profiler.save_results()


if __name__ == "__main__":
    profile_worker_group_startup_with_callbacks()
