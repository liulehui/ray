#!/usr/bin/env python3
"""
Example usage of Ray Train worker group startup profiling tools.

This script demonstrates different ways to profile worker group startup
and analyze the results.
"""

import os
import sys
import time
import ray
from pathlib import Path

# Add the current directory to Python path to import our profiling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker_group_startup_profiler import WorkerGroupStartupProfiler, instrument_worker_group_startup
from worker_group_profiler import setup_ray_train_profiling, profile_worker_group_startup, RayTrainProfilingConfig


def example_1_basic_profiling():
    """
    Example 1: Basic profiling using the focused startup profiler.
    """
    print("=" * 60)
    print("Example 1: Basic Profiling")
    print("=" * 60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create a profiler
    profiler = WorkerGroupStartupProfiler("/tmp/example1_profiling")
    
    # Simple training function
    def simple_train_fn():
        print("Training function started")
        time.sleep(2)  # Simulate some training work
        print("Training function completed")
    
    try:
        from ray.train.v2 import DataParallelTrainer
        
        # Create trainer
        trainer = DataParallelTrainer(
            train_loop_per_worker=simple_train_fn,
            scaling_config={"num_workers": 2, "use_gpu": False},
            run_config={"name": "example1_basic_profiling"}
        )
        
        # Add profiler to callbacks
        original_callbacks = trainer._create_default_callbacks
        def profiled_callbacks():
            callbacks = original_callbacks()
            callbacks.append(profiler)
            return callbacks
        
        trainer._create_default_callbacks = profiled_callbacks
        
        print("Starting training with basic profiling...")
        result = trainer.fit()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save results
        profiler.save_results()
        print(f"Profiling results saved to {profiler.output_dir}")


def example_2_comprehensive_profiling():
    """
    Example 2: Comprehensive profiling with multiple approaches.
    """
    print("\n" + "=" * 60)
    print("Example 2: Comprehensive Profiling")
    print("=" * 60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Setup comprehensive profiling
    config = RayTrainProfilingConfig()
    config.output_dir = "/tmp/example2_comprehensive_profiling"
    config.enable_detailed_profiling = True
    config.enable_cprofile = True
    config.enable_method_instrumentation = True
    
    setup_ray_train_profiling(config)
    
    # Training function with some complexity
    def complex_train_fn():
        print("Complex training function started")
        
        # Simulate model loading
        time.sleep(1)
        print("Model loaded")
        
        # Simulate data loading
        time.sleep(1)
        print("Data loaded")
        
        # Simulate training
        time.sleep(2)
        print("Training completed")
    
    try:
        # Use the comprehensive profiler
        with profile_worker_group_startup("both", config.output_dir) as profilers:
            from ray.train.v2 import DataParallelTrainer
            
            trainer = DataParallelTrainer(
                train_loop_per_worker=complex_train_fn,
                scaling_config={"num_workers": 3, "use_gpu": False},
                run_config={"name": "example2_comprehensive_profiling"}
            )
            
            # Add our detailed profiler to the trainer's callbacks
            if profilers and len(profilers) > 0:
                original_callbacks = trainer._create_default_callbacks
                def profiled_callbacks():
                    callbacks = original_callbacks()
                    # Add the detailed profiler (first one in the list)
                    callbacks.append(profilers[0])
                    return callbacks
                
                trainer._create_default_callbacks = profiled_callbacks
            
            print("Starting training with comprehensive profiling...")
            result = trainer.fit()
            print("Training completed successfully!")
    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Comprehensive profiling results saved to {config.output_dir}")


def example_3_custom_profiling():
    """
    Example 3: Custom profiling with extended phases.
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Profiling")
    print("=" * 60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create a custom profiler that extends the base profiler
    class CustomProfiler(WorkerGroupStartupProfiler):
        def __init__(self, output_dir):
            super().__init__(output_dir)
            self.custom_phases = []
        
        def before_worker_group_start(self, worker_group_context):
            # Add custom phase before the standard profiling
            self._start_phase("custom_pre_startup", {
                "custom_metadata": "This is a custom phase",
                "timestamp": time.time()
            })
            
            # Simulate some custom work
            time.sleep(0.1)
            
            self._end_phase()
            
            # Call the parent method for standard profiling
            super().before_worker_group_start(worker_group_context)
        
        def after_worker_group_start(self, worker_group):
            # Call parent first
            super().after_worker_group_start(worker_group)
            
            # Add custom phase after standard profiling
            self._start_phase("custom_post_startup", {
                "worker_count": len(worker_group),
                "custom_check": "All workers ready"
            })
            
            # Simulate some custom verification
            time.sleep(0.1)
            
            self._end_phase()
    
    # Create custom profiler
    custom_profiler = CustomProfiler("/tmp/example3_custom_profiling")
    
    # Training function
    def custom_train_fn():
        print("Custom training function started")
        time.sleep(1)
        print("Custom training completed")
    
    try:
        from ray.train.v2 import DataParallelTrainer
        
        trainer = DataParallelTrainer(
            train_loop_per_worker=custom_train_fn,
            scaling_config={"num_workers": 2, "use_gpu": False},
            run_config={"name": "example3_custom_profiling"}
        )
        
        # Add custom profiler to callbacks
        original_callbacks = trainer._create_default_callbacks
        def profiled_callbacks():
            callbacks = original_callbacks()
            callbacks.append(custom_profiler)
            return callbacks
        
        trainer._create_default_callbacks = profiled_callbacks
        
        print("Starting training with custom profiling...")
        result = trainer.fit()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save results
        custom_profiler.save_results()
        print(f"Custom profiling results saved to {custom_profiler.output_dir}")


def analyze_results():
    """
    Analyze and display results from all examples.
    """
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)
    
    # Analyze results from each example
    example_dirs = [
        "/tmp/example1_profiling",
        "/tmp/example2_comprehensive_profiling", 
        "/tmp/example3_custom_profiling"
    ]
    
    for i, output_dir in enumerate(example_dirs, 1):
        print(f"\n--- Example {i} Results ---")
        
        # Check if results exist
        if not Path(output_dir).exists():
            print(f"  No results found in {output_dir}")
            continue
        
        # Read summary if available
        summary_file = Path(output_dir) / "startup_summary.txt"
        if summary_file.exists():
            print(f"  Summary from {output_dir}:")
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                # Show first few lines
                for line in lines[:10]:
                    print(f"    {line.rstrip()}")
            print("    ...")
        
        # Read bottleneck analysis if available
        bottleneck_file = Path(output_dir) / "bottleneck_analysis.txt"
        if bottleneck_file.exists():
            print(f"  Bottleneck analysis from {output_dir}:")
            with open(bottleneck_file, 'r') as f:
                lines = f.readlines()
                # Show first few lines
                for line in lines[:8]:
                    print(f"    {line.rstrip()}")
            print("    ...")


def main():
    """
    Run all profiling examples.
    """
    print("Ray Train Worker Group Startup Profiling Examples")
    print("=" * 60)
    print("This script demonstrates different profiling approaches.")
    print("Each example will create a training job and profile its startup.")
    print()
    
    try:
        # Run examples
        example_1_basic_profiling()
        example_2_comprehensive_profiling()
        example_3_custom_profiling()
        
        # Analyze results
        analyze_results()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print("Check the output directories for detailed profiling results:")
        print("  - /tmp/example1_profiling")
        print("  - /tmp/example2_comprehensive_profiling")
        print("  - /tmp/example3_custom_profiling")
        print()
        print("Key files to examine:")
        print("  - startup_summary.txt: Human-readable summary")
        print("  - bottleneck_analysis.txt: Automated analysis and recommendations")
        print("  - startup_phases.json: Detailed timing data")
        print("  - startup_profiling.log: Detailed logs")
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user.")
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main() 