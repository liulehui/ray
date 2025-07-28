#!/usr/bin/env python3
"""
Simple usage example for Ray core operations profiling.

This script demonstrates how to use the Ray core profiler to investigate
bottlenecks in worker.py:2743 (get) and worker.py:2944 (wait) operations.
"""

import ray
from ray_core_profiler import profile_ray_core_operations


def main():
    """
    Main function demonstrating Ray core operations profiling.
    """
    print("Ray Core Operations Profiling Example")
    print("=" * 40)
    print("This will profile the specific bottlenecks you identified:")
    print("- worker.py:2743 (ray.get)")
    print("- worker.py:2944 (ray.wait)")
    print()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Training function that will trigger Ray operations
    def training_function():
        import time

        # This will trigger ray.get operations
        print("Training function started")

        # Simulate some work that might trigger Ray operations
        time.sleep(1)

        # Create some remote objects that will trigger ray.get/ray.wait
        @ray.remote
        def remote_task():
            time.sleep(0.5)
            return "task completed"

        # This will trigger ray.get operations
        results = ray.get([remote_task.remote() for _ in range(4)])
        print(f"Remote tasks completed: {results}")

        print("Training function completed")

    # Profile Ray core operations during worker group startup
    with profile_ray_core_operations("/tmp/ray_core_example_profiling") as profiler:
        try:
            from ray.train.v2 import DataParallelTrainer

            trainer = DataParallelTrainer(
                train_loop_per_worker=training_function,
                scaling_config={"num_workers": 2, "use_gpu": False},
                run_config={"name": "ray_core_profiling_example"},
            )

            print("Starting training with Ray core operation profiling...")
            result = trainer.fit()
            print("Training completed successfully!")

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback

            traceback.print_exc()

    print("\nProfiling results saved to /tmp/ray_core_example_profiling")
    print("\nKey files to examine:")
    print("  - ray_operations.json: Detailed timing for each Ray get/wait operation")
    print(
        "  - ray_operations_summary.txt: Summary with average times and slow operations"
    )
    print(
        "  - ray_bottleneck_analysis.txt: Analysis of bottlenecks and recommendations"
    )
    print("  - ray_operation_patterns.txt: Most common operation patterns")
    print("  - ray_core_profiling.log: Detailed logs of all operations")

    print("\nWhat to look for:")
    print("1. Operations taking > 1 second (slow operations)")
    print("2. Patterns in object counts and timeouts")
    print("3. Stack traces showing where operations are called from")
    print("4. Network-related delays (different node IPs)")
    print("5. Failed operations that might indicate issues")


if __name__ == "__main__":
    main()
