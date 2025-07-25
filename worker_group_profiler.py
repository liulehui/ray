#!/usr/bin/env python3
"""
Comprehensive profiling solution for Ray Train worker group startup.

This module provides multiple profiling approaches to investigate why worker group
startup takes a long time.
"""

import time
import cProfile
import pstats
import io
import functools
import logging
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import ray
from ray.train.v2._internal.execution.worker_group import WorkerGroup
from ray.train.v2._internal.execution.callback import WorkerGroupCallback
from ray.train.v2._internal.callbacks.metrics import ControllerMetricsCallback


@dataclass
class ProfilingResult:
    """Container for profiling results."""
    method_name: str
    duration: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any]


class DetailedWorkerGroupProfiler(WorkerGroupCallback):
    """
    A detailed profiler that instruments the worker group startup process
    with fine-grained timing measurements.
    """
    
    def __init__(self, output_dir: str = "/tmp/ray_train_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiling_results: List[ProfilingResult] = []
        self.current_phase = "unknown"
        self.phase_start_time = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(self.output_dir / "worker_group_profiling.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
    def _record_phase(self, phase_name: str, metadata: Dict[str, Any] = None):
        """Record timing for a specific phase."""
        current_time = time.monotonic()
        
        if self.phase_start_time is not None:
            duration = current_time - self.phase_start_time
            result = ProfilingResult(
                method_name=self.current_phase,
                duration=duration,
                start_time=self.phase_start_time,
                end_time=current_time,
                metadata=metadata or {}
            )
            self.profiling_results.append(result)
            self.logger.info(f"Phase '{self.current_phase}' took {duration:.3f}s")
        
        self.current_phase = phase_name
        self.phase_start_time = current_time
        self.logger.info(f"Starting phase: {phase_name}")
    
    def before_worker_group_start(self, worker_group_context):
        """Called before worker group actors are initialized."""
        self._record_phase("before_worker_group_start", {
            "num_workers": worker_group_context.num_workers,
            "resources_per_worker": worker_group_context.resources_per_worker,
            "placement_strategy": worker_group_context.placement_strategy
        })
    
    def after_worker_group_start(self, worker_group):
        """Called after worker group actors are initialized."""
        self._record_phase("after_worker_group_start", {
            "num_workers": len(worker_group),
            "worker_ips": [w.metadata.node_ip for w in worker_group.get_workers()],
            "worker_pids": [w.metadata.pid for w in worker_group.get_workers()]
        })
    
    def after_worker_group_training_start(self, worker_group):
        """Called after training starts on all workers."""
        self._record_phase("after_worker_group_training_start")
    
    def before_worker_group_shutdown(self, worker_group):
        """Called before worker group shutdown."""
        self._record_phase("before_worker_group_shutdown")
    
    def save_results(self):
        """Save profiling results to file."""
        import json
        
        results_file = self.output_dir / "profiling_results.json"
        with open(results_file, 'w') as f:
            json.dump([{
                "method_name": r.method_name,
                "duration": r.duration,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "metadata": r.metadata
            } for r in self.profiling_results], f, indent=2)
        
        # Generate summary report
        summary_file = self.output_dir / "profiling_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Ray Train Worker Group Startup Profiling Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total_time = sum(r.duration for r in self.profiling_results)
            f.write(f"Total profiled time: {total_time:.3f}s\n\n")
            
            f.write("Phase breakdown:\n")
            for result in self.profiling_results:
                percentage = (result.duration / total_time) * 100 if total_time > 0 else 0
                f.write(f"  {result.method_name}: {result.duration:.3f}s ({percentage:.1f}%)\n")
        
        self.logger.info(f"Profiling results saved to {self.output_dir}")


class CProfileWorkerGroupProfiler:
    """
    Uses Python's cProfile to get detailed function-level profiling.
    """
    
    def __init__(self, output_dir: str = "/tmp/ray_train_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = None
        self.stats_file = self.output_dir / "cprofile_stats.prof"
    
    def start_profiling(self):
        """Start cProfile profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def stop_profiling(self):
        """Stop cProfile profiling and save results."""
        if self.profiler:
            self.profiler.disable()
            self.profiler.dump_stats(str(self.stats_file))
            
            # Generate readable stats
            stats = pstats.Stats(self.profiler)
            stats_file = self.output_dir / "cprofile_stats.txt"
            
            with open(stats_file, 'w') as f:
                # Redirect stdout to capture stats output
                old_stdout = sys.stdout
                sys.stdout = f
                stats.sort_stats('cumulative')
                stats.print_stats(50)  # Top 50 functions
                stats.print_callers(20)  # Top 20 callers
                sys.stdout = old_stdout


@contextmanager
def profile_worker_group_startup(
    profiling_type: str = "detailed",
    output_dir: str = "/tmp/ray_train_profiling"
):
    """
    Context manager for profiling worker group startup.
    
    Args:
        profiling_type: "detailed", "cprofile", or "both"
        output_dir: Directory to save profiling results
    """
    profilers = []
    
    if profiling_type in ["detailed", "both"]:
        detailed_profiler = DetailedWorkerGroupProfiler(output_dir)
        profilers.append(detailed_profiler)
    
    if profiling_type in ["cprofile", "both"]:
        cprofile_profiler = CProfileWorkerGroupProfiler(output_dir)
        profilers.append(cprofile_profiler)
        cprofile_profiler.start_profiling()
    
    try:
        yield profilers
    finally:
        if profiling_type in ["cprofile", "both"]:
            cprofile_profiler.stop_profiling()
        
        if profiling_type in ["detailed", "both"]:
            detailed_profiler.save_results()


def instrument_worker_group_methods():
    """
    Monkey patch WorkerGroup methods to add timing instrumentation.
    This provides method-level timing without modifying the original code.
    """
    original_start_impl = WorkerGroup._start_impl
    original_create_workers = WorkerGroup._create_workers
    original_init_train_context = WorkerGroup._init_train_context_on_workers
    
    def timed_start_impl(self, worker_group_state_builder):
        start_time = time.monotonic()
        print(f"[PROFILING] Starting _start_impl at {start_time}")
        
        try:
            result = original_start_impl(self, worker_group_state_builder)
            end_time = time.monotonic()
            print(f"[PROFILING] _start_impl completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.monotonic()
            print(f"[PROFILING] _start_impl failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    def timed_create_workers(self, num_workers, placement_group, resources_per_worker):
        start_time = time.monotonic()
        print(f"[PROFILING] Starting _create_workers at {start_time}")
        
        try:
            result = original_create_workers(self, num_workers, placement_group, resources_per_worker)
            end_time = time.monotonic()
            print(f"[PROFILING] _create_workers completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.monotonic()
            print(f"[PROFILING] _create_workers failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    def timed_init_train_context(self, workers, sync_actor, train_context_args):
        start_time = time.monotonic()
        print(f"[PROFILING] Starting _init_train_context_on_workers at {start_time}")
        
        try:
            result = original_init_train_context(self, workers, sync_actor, train_context_args)
            end_time = time.monotonic()
            print(f"[PROFILING] _init_train_context_on_workers completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.monotonic()
            print(f"[PROFILING] _init_train_context_on_workers failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    # Apply the monkey patches
    WorkerGroup._start_impl = timed_start_impl
    WorkerGroup._create_workers = timed_create_workers
    WorkerGroup._init_train_context_on_workers = timed_init_train_context


class RayTrainProfilingConfig:
    """
    Configuration class for Ray Train profiling.
    """
    
    def __init__(self):
        self.enable_detailed_profiling = True
        self.enable_cprofile = False
        self.enable_method_instrumentation = True
        self.output_dir = "/tmp/ray_train_profiling"
        self.log_level = logging.DEBUG


def setup_ray_train_profiling(config: RayTrainProfilingConfig = None):
    """
    Setup comprehensive profiling for Ray Train worker group startup.
    
    Args:
        config: Profiling configuration
    """
    if config is None:
        config = RayTrainProfilingConfig()
    
    # Setup logging
    logging.basicConfig(
        level=config.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.output_dir}/ray_train_profiling.log"),
            logging.StreamHandler()
        ]
    )
    
    # Apply method instrumentation if enabled
    if config.enable_method_instrumentation:
        instrument_worker_group_methods()
        logging.info("Applied method instrumentation to WorkerGroup")
    
    return config


# Example usage functions
def profile_simple_training_run():
    """
    Example of how to profile a simple training run.
    """
    from ray.train.v2 import DataParallelTrainer
    from ray.train.v2._internal.callbacks.metrics import ControllerMetricsCallback
    
    def simple_train_fn():
        import time
        time.sleep(1)  # Simulate some training work
    
    # Setup profiling
    config = RayTrainProfilingConfig()
    config.output_dir = "/tmp/ray_train_simple_profiling"
    setup_ray_train_profiling(config)
    
    # Create trainer with profiling callbacks
    with profile_worker_group_startup("both", config.output_dir) as profilers:
        trainer = DataParallelTrainer(
            train_loop_per_worker=simple_train_fn,
            scaling_config={"num_workers": 2, "use_gpu": False},
            run_config={"name": "profiling_test"}
        )
        
        # Add our detailed profiler to the trainer's callbacks
        if profilers and hasattr(profilers[0], '__class__') and 'DetailedWorkerGroupProfiler' in str(profilers[0].__class__):
            trainer._create_default_callbacks = lambda: [profilers[0]]
        
        result = trainer.fit()
    
    print(f"Training completed. Check profiling results in {config.output_dir}")


def analyze_profiling_results(output_dir: str):
    """
    Analyze and display profiling results.
    
    Args:
        output_dir: Directory containing profiling results
    """
    output_path = Path(output_dir)
    
    # Read detailed profiling results
    results_file = output_path / "profiling_results.json"
    if results_file.exists():
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("\n=== Detailed Profiling Results ===")
        total_time = sum(r['duration'] for r in results)
        print(f"Total profiled time: {total_time:.3f}s")
        
        for result in results:
            percentage = (result['duration'] / total_time) * 100 if total_time > 0 else 0
            print(f"  {result['method_name']}: {result['duration']:.3f}s ({percentage:.1f}%)")
    
    # Read cProfile results
    stats_file = output_path / "cprofile_stats.txt"
    if stats_file.exists():
        print("\n=== cProfile Results (Top 10 functions) ===")
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            # Find and display the top functions
            for i, line in enumerate(lines):
                if "ncalls" in line and "tottime" in line:
                    print("".join(lines[i:i+15]))  # Show header + 10 functions
                    break


if __name__ == "__main__":
    # Example usage
    print("Ray Train Worker Group Profiling Tool")
    print("=" * 40)
    
    # Setup profiling
    config = RayTrainProfilingConfig()
    config.output_dir = "/tmp/ray_train_demo_profiling"
    setup_ray_train_profiling(config)
    
    # Run a simple profiling example
    profile_simple_training_run()
    
    # Analyze results
    analyze_profiling_results(config.output_dir) 