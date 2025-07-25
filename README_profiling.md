# Ray Train Worker Group Startup Profiling

This directory contains comprehensive profiling tools to investigate why Ray Train worker group startup takes a long time.

## Overview

The profiling tools provide multiple approaches to identify bottlenecks in the worker group startup process:

1. **Detailed Phase Profiling**: Tracks each phase of the startup process with timing and metadata
2. **Method-Level Instrumentation**: Adds timing to specific WorkerGroup methods
3. **cProfile Integration**: Provides function-level profiling using Python's cProfile
4. **Bottleneck Analysis**: Automatically identifies slow phases and provides recommendations

## Files

- `worker_group_profiler.py`: Comprehensive profiling solution with multiple approaches
- `worker_group_startup_profiler.py`: Focused profiling specifically for startup bottlenecks
- `README_profiling.md`: This documentation file

## Quick Start

### 1. Basic Profiling

```python
from worker_group_startup_profiler import profile_worker_group_startup_bottlenecks

# Run the profiler
profile_worker_group_startup_bottlenecks()
```

### 2. Using the Comprehensive Profiler

```python
from worker_group_profiler import (
    setup_ray_train_profiling,
    profile_worker_group_startup,
    RayTrainProfilingConfig
)

# Setup profiling configuration
config = RayTrainProfilingConfig()
config.output_dir = "/tmp/my_profiling_results"
config.enable_detailed_profiling = True
config.enable_cprofile = True
config.enable_method_instrumentation = True

# Setup the profiling
setup_ray_train_profiling(config)

# Use in your training code
with profile_worker_group_startup("both", config.output_dir) as profilers:
    from ray.train.v2 import DataParallelTrainer

    def my_train_fn():
        # Your training function
        pass

    trainer = DataParallelTrainer(
        train_loop_per_worker=my_train_fn,
        scaling_config={"num_workers": 4, "use_gpu": True},
        run_config={"name": "profiled_training"}
    )

    result = trainer.fit()
```

### 3. Manual Integration

```python
from worker_group_startup_profiler import WorkerGroupStartupProfiler

# Create a profiler
profiler = WorkerGroupStartupProfiler("/tmp/custom_profiling")

# Add to your trainer's callbacks
from ray.train.v2 import DataParallelTrainer

trainer = DataParallelTrainer(
    train_loop_per_worker=my_train_fn,
    scaling_config={"num_workers": 2, "use_gpu": False}
)

# Modify the trainer's callback creation
original_callbacks = trainer._create_default_callbacks
def profiled_callbacks():
    callbacks = original_callbacks()
    callbacks.append(profiler)
    return callbacks

trainer._create_default_callbacks = profiled_callbacks

# Run training
result = trainer.fit()

# Save results
profiler.save_results()
```

## Understanding the Results

### Output Files

After running the profiler, you'll find several files in the output directory:

1. **`startup_phases.json`**: Detailed timing for each phase
2. **`startup_summary.txt`**: Human-readable summary
3. **`bottleneck_analysis.txt`**: Automated bottleneck analysis and recommendations
4. **`startup_profiling.log`**: Detailed logs
5. **`cprofile_stats.prof`**: cProfile binary data (if enabled)
6. **`cprofile_stats.txt`**: Readable cProfile results (if enabled)

### Sample Output

```
Ray Train Worker Group Startup Summary
========================================

Total startup time: 15.234s

Phase breakdown:
  before_worker_group_start: 0.001s (0.0%)
    Metadata: {'num_workers': 4, 'resources_per_worker': {'CPU': 1, 'GPU': 1}}

  placement_group_creation: 8.456s (55.5%)
    Metadata: {'placement_strategy': 'PACK'}

  actor_creation: 3.234s (21.2%)
    Metadata: {'num_workers': 4}

  context_initialization: 2.123s (13.9%)
    Metadata: {'num_workers': 4}

  after_worker_group_start: 1.420s (9.3%)
    Metadata: {'num_workers': 4, 'worker_ips': ['192.168.1.10', '192.168.1.11']}
```

### Bottleneck Analysis

The profiler automatically identifies slow phases and provides recommendations:

```
Worker Group Startup Bottleneck Analysis
========================================

Slowest phase: placement_group_creation (8.456s)

SLOW PHASE DETECTED: placement_group_creation took 8.456s
Recommendations:
  - Check cluster resource availability
  - Consider reducing worker count or resource requirements
  - Check if placement group strategy is optimal
```

## Common Bottlenecks and Solutions

### 1. Placement Group Creation (Most Common)

**Symptoms**: Long wait times for placement group to be ready
**Causes**:
- Insufficient cluster resources
- Resource fragmentation
- Network issues between nodes

**Solutions**:
- Check cluster resource availability: `ray status`
- Reduce worker count or resource requirements
- Use different placement strategy (e.g., `SPREAD` instead of `PACK`)
- Add more nodes to the cluster

### 2. Actor Creation and Initialization

**Symptoms**: Slow worker process startup
**Causes**:
- Large runtime environment (pip packages, conda environments)
- Slow network for downloading dependencies
- Resource contention on nodes

**Solutions**:
- Optimize runtime environment (remove unnecessary packages)
- Use pre-built container images
- Check worker startup logs for errors
- Ensure sufficient CPU/memory on worker nodes

### 3. Context Initialization

**Symptoms**: Slow training context setup
**Causes**:
- Large model loading
- Dataset preprocessing
- Slow storage access

**Solutions**:
- Use model checkpointing to avoid reloading
- Optimize dataset loading and preprocessing
- Use faster storage (SSD vs HDD)
- Pre-warm data loading

### 4. Network Latency

**Symptoms**: Workers on different nodes taking longer to start
**Causes**:
- Cross-node communication overhead
- Network bandwidth limitations

**Solutions**:
- Use node affinity to place workers on same node
- Optimize network configuration
- Use placement group strategies that minimize cross-node communication

## Advanced Usage

### Custom Profiling Phases

You can extend the profiler to track custom phases:

```python
class CustomProfiler(WorkerGroupStartupProfiler):
    def before_worker_group_start(self, worker_group_context):
        self._start_phase("custom_pre_startup")
        # Your custom logic
        self._end_phase()
        super().before_worker_group_start(worker_group_context)
```

### Integration with Ray Dashboard

The profiling results can be correlated with Ray Dashboard metrics:

1. Check the Ray Dashboard for cluster resource usage
2. Look at worker process logs in the dashboard
3. Monitor network and disk I/O during startup

### Environment Variables

You can control profiling behavior with environment variables:

```bash
export RAY_TRAIN_PROFILING_OUTPUT_DIR="/custom/path"
export RAY_TRAIN_PROFILING_LOG_LEVEL="DEBUG"
```

## Troubleshooting

### Profiler Not Working

1. Ensure Ray is properly initialized: `ray.init()`
2. Check that the profiler is added to trainer callbacks
3. Verify output directory permissions

### Missing Data

1. Check if training completed successfully
2. Ensure `save_results()` is called
3. Look for errors in the profiling log

### Inaccurate Timing

1. Ensure system clock is synchronized
2. Check for system load during profiling
3. Run multiple times to get average measurements

## Performance Benchmarks

Typical startup times for reference:

- **Small cluster (2-4 workers, CPU only)**: 2-5 seconds
- **Medium cluster (4-8 workers, with GPU)**: 5-15 seconds
- **Large cluster (8+ workers, multi-GPU)**: 10-30 seconds

Times exceeding these ranges indicate potential bottlenecks that should be investigated.

## Contributing

To add new profiling capabilities:

1. Extend the `WorkerGroupStartupProfiler` class
2. Add new phases to track specific operations
3. Update the bottleneck analysis logic
4. Add tests for new functionality

## Support

For issues with the profiling tools:

1. Check the profiling logs for errors
2. Verify Ray Train version compatibility
3. Report issues with detailed reproduction steps
4. Include profiling output files when reporting bugs
