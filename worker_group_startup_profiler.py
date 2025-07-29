#!/usr/bin/env python3
"""
Focused profiling for Ray Train worker group startup bottlenecks.

This script provides targeted profiling to identify specific bottlenecks
in the worker group startup process.
"""

import os

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import time
import logging
import traceback
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import ray

# from ray.train.v2._internal.execution.worker_group import WorkerGroup
# from ray.train.v2._internal.execution.callback import WorkerGroupCallback
from ray.train.v2._internal.util import get_callable_name
from worker_group_startup_profiler_v2 import profile_worker_group_startup_v2
from combined_worker_group_profiler import profile_worker_group_combined
from ray_core_profiler import profile_ray_core_operations


from ray.train import CheckpointConfig, Result, RunConfig, ScalingConfig

def simple_train_fn():
        import time

        print("Training function started")
        time.sleep(2)
        print("Training function completed")

def profile_worker_group_startup_bottlenecks():
    """
    Main function to profile worker group startup bottlenecks.
    """
    print("Ray Train Worker Group Startup Bottleneck Profiler")
    print("=" * 50)

    from ray.train.xgboost import XGBoostTrainer

    # with profile_worker_group_startup_v2("/Users/lehui/Desktop/Anyscale/profiling_wg_startup/0728/") as profiler:
    with profile_ray_core_operations("/Users/lehui/Desktop/Anyscale/profiling_wg_startup/0728/raycore/") as profiler:
    
        run_config = ray.train.RunConfig(
            checkpoint_config=ray.train.CheckpointConfig(
                # Checkpoint every 10 iterations.
                checkpoint_frequency=10,
                # Only keep the latest checkpoint.
                num_to_keep=1,
            ),
            callbacks=[profiler]
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

        print("Starting training run to profile worker group startup...")
        result = ray_xgbooster_trainer.fit()

        print("Training completed successfully!")
        print(f"Ray train result:", result)


if __name__ == "__main__":
    profile_worker_group_startup_bottlenecks()
