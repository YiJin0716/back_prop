"""Epoch-based cosine schedules for diagnostic temperature and learning rates."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TemperatureSchedule:
    start: float = 1.0
    end: float = 0.25
    epochs: int = 18

    def __post_init__(self):
        if not (math.isfinite(self.start) and math.isfinite(self.end)
                and 0 < self.end <= self.start and self.epochs >= 2):
            raise ValueError("Require finite 0 < end <= start and at least two epochs")

    def __call__(self, epoch):
        """Zero-based epoch: first=start, last=end; clamp outside the horizon."""
        progress = min(max(float(epoch) / (self.epochs - 1), 0.0), 1.0)
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        return {"kind": "cosine_by_epoch", **asdict(self)}


@dataclass(frozen=True)
class LearningRateSchedule:
    """Common LR multiplier with a per-group floor and no cosine restart.

    Passed to torch.optim.lr_scheduler.LambdaLR. Epoch zero uses the base LR;
    epoch epochs-1 uses min_ratio * base LR. Extra steps stay at the floor.
    The horizon is independent of short tests and early checkpoint exits.
    """

    min_ratio: float = 0.1
    epochs: int = 18

    def __post_init__(self):
        if not (math.isfinite(self.min_ratio) and 0 < self.min_ratio < 1 and self.epochs >= 2):
            raise ValueError("Require finite 0 < lr_min_ratio < 1 and at least two epochs")

    def __call__(self, epoch):
        progress = min(max(float(epoch) / (self.epochs - 1), 0.0), 1.0)
        return self.min_ratio + 0.5 * (1 - self.min_ratio) * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        return {"kind": "cosine_by_epoch", **asdict(self)}


def assert_learning_rates(optimizer, base_lrs, factor):
    names = [group['name'] for group in optimizer.param_groups]
    if len(names) != len(base_lrs) or set(names) != set(base_lrs):
        raise ValueError(f"Optimizer groups differ from the LR schedule: {names}")
    for group in optimizer.param_groups:
        expected = base_lrs[group['name']] * factor
        if not math.isclose(group['lr'], expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"Unexpected LR for {group['name']}: {group['lr']} != {expected}")


def make_lr_scheduler(optimizer, schedule, base_lrs, *, next_epoch=0, saved_state=None):
    """Restore the next epoch's LR without restarting or compounding decay."""
    from torch.optim.lr_scheduler import LambdaLR

    if next_epoch < 0:
        raise ValueError("next_epoch must be nonnegative")
    if saved_state is not None:
        if saved_state['last_epoch'] != next_epoch:
            raise ValueError("LR scheduler epoch differs from checkpoint progress")
        if saved_state['base_lrs'] != [base_lrs[g['name']] for g in optimizer.param_groups]:
            raise ValueError("LR scheduler base learning rates differ from the checkpoint")
        if saved_state.get('lr_lambdas') != [asdict(schedule)] * len(optimizer.param_groups):
            raise ValueError("Saved LR lambda parameters differ from the configured schedule")
        assert_learning_rates(optimizer, base_lrs, schedule(next_epoch))
    for group in optimizer.param_groups:
        group['initial_lr'] = base_lrs[group['name']]
    scheduler = LambdaLR(optimizer, schedule, last_epoch=next_epoch - 1)
    if saved_state is not None:
        scheduler.load_state_dict(saved_state)
    assert_learning_rates(optimizer, base_lrs, schedule(next_epoch))
    if any(not math.isclose(a, g['lr'], rel_tol=1e-12, abs_tol=0)
           for a, g in zip(scheduler.get_last_lr(), optimizer.param_groups)):
        raise ValueError("LR scheduler state disagrees with optimizer learning rates")
    return scheduler
