"""Conservative stopping rule for continued, independently trained warmups.

These are optimization probes on the training cohort, not held-out validation.
Keep the best qualifying checkpoint; an epoch cap is never convergence.
"""
from dataclasses import asdict, dataclass, field
import math


@dataclass
class PlateauStop:
    minimum_epochs: int = 20
    patience: int = 12
    lr_patience: int = 4
    relative_delta: float = .002
    training_delta: float = .005
    best: float | None = None
    bad_epochs: int = 0
    lr_bad_epochs: int = 0
    reductions: int = 0
    since_reduction: int = 0
    completed: int = 0
    passing: int = 0
    training: list[float] = field(default_factory=list)

    def __post_init__(self):
        if min(self.minimum_epochs, self.patience, self.lr_patience) < 1:
            raise ValueError('Convergence durations must be positive')
        if not 0 < self.relative_delta < 1 or not 0 < self.training_delta < 1:
            raise ValueError('Convergence tolerances must be in (0,1)')

    def baseline(self, loss):
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('Invalid convergence baseline')
        self.best = loss

    def update(self, probe_loss, training_loss, passed):
        if not all(math.isfinite(x) and x >= 0 for x in (probe_loss, training_loss)):
            raise ValueError('Invalid convergence loss')
        self.completed += 1
        self.since_reduction += 1
        self.passing = self.passing + 1 if passed else 0
        improved = self.best is None or probe_loss < self.best * (1-self.relative_delta)
        if improved:
            self.best = probe_loss
            self.bad_epochs = self.lr_bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.lr_bad_epochs += 1
        reduce_lr = self.lr_bad_epochs >= self.lr_patience and self.reductions < 3
        if reduce_lr:
            self.reductions += 1
            self.lr_bad_epochs = self.since_reduction = 0
        self.training = (self.training + [training_loss])[-10:]
        gain = None
        if len(self.training) == 10:
            before, after = sum(self.training[:5])/5, sum(self.training[5:])/5
            gain = (before-after)/max(before, 1e-12)
        # A worsening probe must not be labeled convergence merely because it
        # stopped improving. A failed gate likewise cannot release joint training.
        stable = self.best is not None and probe_loss <= self.best * 1.02
        converged = (self.completed >= self.minimum_epochs and self.bad_epochs >= self.patience
            and self.reductions >= 2 and self.since_reduction >= self.lr_patience
            and self.passing >= 3 and gain is not None
            and -0.02 <= gain <= self.training_delta and stable)
        return dict(converged=converged, reduce_lr=reduce_lr, meaningful_improvement=improved,
                    training_relative_gain=gain, lr_ratio=.5**self.reductions,
                    bad_epochs=self.bad_epochs, reductions=self.reductions,
                    additional_epochs=self.completed, passing=self.passing)

    def state_dict(self):
        return asdict(self)
