import numbers
from typing import Dict, Optional

from ray.train.v2._internal.util import requires_train_worker
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@requires_train_worker()
def report(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Report health metrics from the training function.

    Unlike ``ray.train.report``, this is not a barrier: each worker can call it
    any number of times. The metrics are merged into what this worker reported
    before, keeping the latest value of each, and the controller reads them on
    its next poll of the worker.

    Example:

        .. testcode::
            :skipif: True

            import ray.train.health as health

            def train_func(config):
                for step, batch in enumerate(loader):
                    ...
                    health.report({"grad_norm": grad_norm}, step=step)

    Args:
        metrics: Numeric metrics by name.
        step: The training step the metrics belong to. ``None`` keeps the step
            reported before.

    Raises:
        TypeError: If a metric name is not a string or a value is not a number.
    """
    for name, value in metrics.items():
        if not isinstance(name, str) or not isinstance(value, numbers.Real):
            raise TypeError(
                "Health metrics must map strings to numbers, got "
                f"{name!r}: {type(value).__name__}."
            )
    # Avoid a circular import: Ray Train internals import ray.train.health.
    from ray.train.v2._internal.execution.train_fn_utils import get_train_fn_utils

    get_train_fn_utils().report_health(dict(metrics), step)
