import math
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, milestones, min_ratio=0., cycle_decay=1., warmup_iters=1000, warmup_factor=1./10, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )
        self.milestones = [warmup_iters]+milestones
        self.min_ratio = min_ratio
        self.cycle_decay = cycle_decay
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # which cyle is it
            cycle = min(bisect_right(self.milestones, self.last_epoch), len(self.milestones)-1)
            # calculate the fraction in the cycle
            fraction = min((self.last_epoch - self.milestones[cycle-1]) / (self.milestones[cycle]-self.milestones[cycle-1]), 1.)

            return [base_lr*self.min_ratio + (base_lr * self.cycle_decay**(cycle-1) - base_lr*self.min_ratio) *
                    (1 + math.cos(math.pi * fraction)) / 2
                    for base_lr in self.base_lrs]