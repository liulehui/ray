#!/usr/bin/env python3
"""
Focused profiling for Ray Train worker group startup bottlenecks.

This script provides targeted profiling to identify specific bottlenecks
in the worker group startup process.
"""

import os

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

print(os.environ)

import time
import logging
import traceback
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import ray

from ray.train.v2._internal.execution.worker_group import WorkerGroup
from ray.train.v2._internal.execution.callback import WorkerGroupCallback
from ray.train.v2._internal.util import get_callable_name


from ray.train import ScalingConfig


@dataclass
class StartupPhase:
    """Represents a phase in the worker group startup process."""

    name: str
    start_time: float
    end_time: float = None
    duration: float = None
    metadata: Dict[str, Any] = None
    error: str = None

    def finish(self, metadata: Dict[str, Any] = None):
        """Mark the phase as finished and calculate duration."""
        self.end_time = time.monotonic()
        self.duration = self.end_time - self.start_time
        if metadata:
            self.metadata = metadata


class WorkerGroupStartupProfiler(WorkerGroupCallback):
    """
    Profiler specifically designed to identify worker group startup bottlenecks.
    """

    def __init__(self, output_dir: str = "/tmp/worker_group_startup_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Track phases
        self.phases: List[StartupPhase] = []
        self.current_phase: StartupPhase = None

        # Track specific bottlenecks
        self.placement_group_time = 0
        self.actor_creation_time = 0
        self.actor_initialization_time = 0
        self.context_initialization_time = 0

    def _setup_logging(self):
        """Setup detailed logging for the profiler."""
        logger = logging.getLogger("WorkerGroupStartupProfiler")
        logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.output_dir / "startup_profiling.log")
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

    def _start_phase(self, phase_name: str, metadata: Dict[str, Any] = None):
        """Start a new phase."""
        if self.current_phase:
            self._end_phase()

        self.current_phase = StartupPhase(
            name=phase_name, start_time=time.monotonic(), metadata=metadata or {}
        )
        self.phases.append(self.current_phase)
        self.logger.info(f"Starting phase: {phase_name}")

    def _end_phase(self, metadata: Dict[str, Any] = None):
        """End the current phase."""
        if self.current_phase:
            self.current_phase.finish(metadata)
            self.logger.info(
                f"Phase '{self.current_phase.name}' completed in {self.current_phase.duration:.3f}s"
            )
            self.current_phase = None

    def _add_metadata_to_current_phase(self, metadata: Dict[str, Any]):
        """Add metadata to the current phase."""
        if self.current_phase:
            if self.current_phase.metadata is None:
                self.current_phase.metadata = {}
            self.current_phase.metadata.update(metadata)

    def before_worker_group_start(self, worker_group_context):
        """Called before worker group actors are initialized."""
        self._start_phase(
            "before_worker_group_start",
            {
                "num_workers": worker_group_context.num_workers,
                "resources_per_worker": worker_group_context.resources_per_worker,
                "placement_strategy": worker_group_context.placement_strategy,
            },
        )

    def after_worker_group_start(self, worker_group):
        """Called after worker group actors are initialized."""
        self._add_metadata_to_current_phase(
            {
                "num_workers": len(worker_group),
                "worker_ips": [w.metadata.node_ip for w in worker_group.get_workers()],
                "worker_pids": [w.metadata.pid for w in worker_group.get_workers()],
            }
        )
        self._end_phase()

    def after_worker_group_training_start(self, worker_group):
        """Called after training starts on all workers."""
        self._start_phase("after_worker_group_training_start")
        self._end_phase()

    def before_worker_group_shutdown(self, worker_group):
        """Called before worker group shutdown."""
        self._start_phase("before_worker_group_shutdown")
        self._end_phase()

    def save_results(self):
        """Save detailed profiling results."""
        # Save phases to JSON
        phases_data = [asdict(phase) for phase in self.phases]
        with open(self.output_dir / "startup_phases.json", "w") as f:
            json.dump(phases_data, f, indent=2)

        # Generate summary report
        self._generate_summary_report()

        # Generate bottleneck analysis
        self._generate_bottleneck_analysis()

        self.logger.info(f"Profiling results saved to {self.output_dir}")

    def _generate_summary_report(self):
        """Generate a summary report of the startup process."""
        with open(self.output_dir / "startup_summary.txt", "w") as f:
            f.write("Ray Train Worker Group Startup Summary\n")
            f.write("=" * 40 + "\n\n")

            total_time = sum(phase.duration for phase in self.phases if phase.duration)
            f.write(f"Total startup time: {total_time:.3f}s\n\n")

            f.write("Phase breakdown:\n")
            for phase in self.phases:
                if phase.duration:
                    percentage = (
                        (phase.duration / total_time) * 100 if total_time > 0 else 0
                    )
                    f.write(
                        f"  {phase.name}: {phase.duration:.3f}s ({percentage:.1f}%)\n"
                    )
                    if phase.metadata:
                        f.write(f"    Metadata: {phase.metadata}\n")
                    if phase.error:
                        f.write(f"    Error: {phase.error}\n")
                    f.write("\n")

    def _generate_bottleneck_analysis(self):
        """Generate bottleneck analysis and recommendations."""
        with open(self.output_dir / "bottleneck_analysis.txt", "w") as f:
            f.write("Worker Group Startup Bottleneck Analysis\n")
            f.write("=" * 40 + "\n\n")

            # Find the slowest phase
            if self.phases:
                slowest_phase = max(self.phases, key=lambda p: p.duration or 0)
                f.write(
                    f"Slowest phase: {slowest_phase.name} ({slowest_phase.duration:.3f}s)\n\n"
                )

            # Common bottleneck patterns
            f.write("Common bottleneck patterns to check:\n")
            f.write(
                "1. Placement Group Creation: Check if cluster has enough resources\n"
            )
            f.write(
                "2. Actor Creation: Check if worker processes are starting slowly\n"
            )
            f.write(
                "3. Context Initialization: Check if training context setup is slow\n"
            )
            f.write("4. Network Latency: Check if workers are on different nodes\n")
            f.write(
                "5. Resource Contention: Check if other jobs are using resources\n\n"
            )

            # Recommendations based on phases
            for phase in self.phases:
                if phase.duration and phase.duration > 5.0:  # Flag phases taking >5s
                    f.write(
                        f"SLOW PHASE DETECTED: {phase.name} took {phase.duration:.3f}s\n"
                    )
                    f.write("Recommendations:\n")

                    if "placement_group" in phase.name.lower():
                        f.write("  - Check cluster resource availability\n")
                        f.write(
                            "  - Consider reducing worker count or resource requirements\n"
                        )
                        f.write("  - Check if placement group strategy is optimal\n")

                    elif "actor" in phase.name.lower():
                        f.write("  - Check worker process startup logs\n")
                        f.write("  - Verify runtime environment setup\n")
                        f.write("  - Check if dependencies are being installed\n")

                    elif "context" in phase.name.lower():
                        f.write("  - Check training context initialization\n")
                        f.write("  - Verify dataset loading and preprocessing\n")
                        f.write("  - Check if model loading is slow\n")

                    f.write("\n")


def instrument_worker_group_startup():
    """
    Instrument the WorkerGroup startup methods to add detailed timing.
    """
    print(">>> try to override the original wg start_impl, create_worker etc...")
    original_start_impl = WorkerGroup._start_impl
    original_create_workers = WorkerGroup._create_workers
    original_init_train_context = WorkerGroup._init_train_context_on_workers
    print(
        ">>> get callable name original start impl:",
        get_callable_name(original_start_impl),
    )

    def instrumented_start_impl(self, worker_group_state_builder):
        """Instrumented version of _start_impl with detailed timing."""
        start_time = time.monotonic()
        print("[STARTUP_PROFILING] Starting _start_impl")

        try:
            # Phase 1: Placement group creation
            pg_start = time.monotonic()
            worker_group_context = self._worker_group_context

            pg = ray.util.placement_group(
                bundles=[worker_group_context.resources_per_worker]
                * worker_group_context.num_workers,
                strategy=worker_group_context.placement_strategy,
            )

            # Wait for placement group
            try:
                ray.get(pg.ready(), timeout=self._worker_group_start_timeout_s)
                pg_end = time.monotonic()
                print(
                    f"[STARTUP_PROFILING] Placement group ready in {pg_end - pg_start:.3f}s"
                )
            except Exception as e:
                pg_end = time.monotonic()
                print(
                    f"[STARTUP_PROFILING] Placement group failed after {pg_end - pg_start:.3f}s: {e}"
                )
                raise

            # Phase 2: Sync actor creation
            sync_start = time.monotonic()
            sync_actor = ray.remote(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote()
            sync_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Sync actor created in {sync_end - sync_start:.3f}s"
            )

            # Phase 3: Worker creation
            workers_start = time.monotonic()
            workers = self._create_workers(
                worker_group_context.num_workers,
                pg,
                worker_group_context.resources_per_worker,
            )
            workers_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Workers created in {workers_end - workers_start:.3f}s"
            )

            # Phase 4: Context initialization
            context_start = time.monotonic()
            train_context_args = {}
            for callable in self._callbacks:
                args = callable.before_init_train_context(workers)
                for arg, arg_values in args.items():
                    train_context_args[arg] = arg_values

            self._init_train_context_on_workers(workers, sync_actor, train_context_args)
            context_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Context initialized in {context_end - context_start:.3f}s"
            )

            # Phase 5: Final setup
            final_start = time.monotonic()
            self._worker_group_state = worker_group_state_builder.build()

            for callback in self._callbacks:
                callback.after_worker_group_start(self)
            final_end = time.monotonic()
            print(f"[STARTUP_PROFILING] Final setup in {final_end - final_start:.3f}s")

            # Launch training function
            launch_start = time.monotonic()
            ray.get(
                [
                    worker.actor.run_train_fn.remote(worker_group_context.train_fn_ref)
                    for worker in workers
                ]
            )
            launch_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Training launched in {launch_end - launch_start:.3f}s"
            )

            for callback in self._callbacks:
                callback.after_worker_group_training_start(self)

            end_time = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Total startup time: {end_time - start_time:.3f}s"
            )

            return None  # Original method doesn't return anything

        except Exception as e:
            end_time = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Startup failed after {end_time - start_time:.3f}s: {e}"
            )
            raise

    def instrumented_create_workers(
        self, num_workers, placement_group, resources_per_worker
    ):
        """Instrumented version of _create_workers with detailed timing."""
        start_time = time.monotonic()
        print(f"[STARTUP_PROFILING] Starting worker creation for {num_workers} workers")

        try:
            # Phase 1: Runtime environment setup
            env_start = time.monotonic()
            runtime_env = self._get_worker_runtime_env(
                custom_runtime_env=self._train_run_context.run_config.worker_runtime_env
            )
            env_end = time.monotonic()
            print(f"[STARTUP_PROFILING] Runtime env setup: {env_end - env_start:.3f}s")

            # Phase 2: Actor class definition
            class_start = time.monotonic()
            worker_actor_cls = ray.remote(
                runtime_env=runtime_env,
                **ray.train.v2._internal.util.bundle_to_remote_args(
                    resources_per_worker
                ),
            )(self._worker_cls)
            class_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Actor class definition: {class_end - class_start:.3f}s"
            )

            # Phase 3: Actor creation
            actors_start = time.monotonic()
            actors = [
                worker_actor_cls.options(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=placement_group, placement_group_bundle_index=i
                    ),
                ).remote()
                for i in range(num_workers)
            ]
            actors_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Actor creation: {actors_end - actors_start:.3f}s"
            )

            # Phase 4: Actor initialization
            init_start = time.monotonic()
            actor_metadatas = ray.get([actor.get_metadata.remote() for actor in actors])
            init_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Actor initialization: {init_end - init_start:.3f}s"
            )

            # Phase 5: Worker object creation
            worker_start = time.monotonic()
            workers = [
                ray.train.v2._internal.execution.worker_group.Worker(
                    actor, meta, resources_per_worker
                )
                for actor, meta in zip(actors, actor_metadatas)
            ]
            workers = WorkerGroup._assign_worker_ranks(workers)
            worker_end = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Worker object creation: {worker_end - worker_start:.3f}s"
            )

            end_time = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Total worker creation time: {end_time - start_time:.3f}s"
            )

            return workers

        except Exception as e:
            end_time = time.monotonic()
            print(
                f"[STARTUP_PROFILING] Worker creation failed after {end_time - start_time:.3f}s: {e}"
            )
            raise

    # Apply the instrumented versions
    WorkerGroup._start_impl = instrumented_start_impl
    WorkerGroup._create_workers = instrumented_create_workers
    print(
        ">>> get callable name instrumented start impl:",
        get_callable_name(instrumented_start_impl),
    )


def profile_worker_group_startup_bottlenecks():
    """
    Main function to profile worker group startup bottlenecks.
    """
    print("Ray Train Worker Group Startup Bottleneck Profiler")
    print("=" * 50)

    # Setup profiling
    profiler = WorkerGroupStartupProfiler(
        "/Users/lehui/Desktop/Anyscale/profiling_wg_startup/"
    )
    instrument_worker_group_startup()

    # Example training function
    def simple_train_fn():
        import time

        time.sleep(1)

    try:
        from ray.train.xgboost import XGBoostTrainer

        run_config = ray.train.RunConfig(
            checkpoint_config=ray.train.CheckpointConfig(
                # Checkpoint every 10 iterations.
                checkpoint_frequency=10,
                # Only keep the latest checkpoint.
                num_to_keep=1,
            ),
            callbacks=[profiler],
        )

        ray_xgbooster_trainer = XGBoostTrainer(
            simple_train_fn,
            train_loop_config={
                "xgboost_params": {
                    "objective": "binary:logistic",
                    "eval_metric": ["logloss", "error"],
                }
            },
            scaling_config=ScalingConfig(
                # Number of workers for data parallelism.
                num_workers=2,
                # Set to True to use GPU acceleration.
                use_gpu=False,
            ),
            run_config=run_config,
        )

        # result: Result = ray_xgbooster_trainer.fit()
        # print(f"Ray train result:", result)

        # Add our profiler to the callbacks
        # original_callbacks = ray_xgbooster_trainer._create_default_callbacks

        # def profiled_callbacks():
        #     callbacks = original_callbacks()
        #     callbacks.append(profiler)
        #     return callbacks

        # ray_xgbooster_trainer._create_default_callbacks = profiled_callbacks

        print("Starting training run to profile worker group startup...")
        result = ray_xgbooster_trainer.fit()

        print("Training completed successfully!")
        print("Ray train result:", result)

    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()

    finally:
        # Save profiling results
        profiler.save_results()
        print(f"\nProfiling results saved to {profiler.output_dir}")
        print("Check the following files for detailed analysis:")
        print("  - startup_phases.json: Detailed phase timing")
        print("  - startup_summary.txt: Summary report")
        print("  - bottleneck_analysis.txt: Bottleneck analysis and recommendations")
        print("  - startup_profiling.log: Detailed logs")


if __name__ == "__main__":
    profile_worker_group_startup_bottlenecks()
