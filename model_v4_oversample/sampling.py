"""One ordinary pass plus bounded training-only hard-case repeats across DDP."""
import hashlib
import json
import math
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Sampler


class MissingNoduleSampler(Sampler):
    def __init__(self, case_ids, *, rank=0, world_size=1, seed=42, fraction=.25,
                 max_extra_per_case=2, history_size=3):
        if len(set(case_ids)) != len(case_ids) or not case_ids:
            raise ValueError('Expected unique training CT IDs')
        if not 0 <= fraction <= 1 or max_extra_per_case < 1 or history_size < 1:
            raise ValueError('Invalid hard-case sampling settings')
        self.case_ids = list(case_ids)
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.fraction, self.max_extra_per_case = fraction, max_extra_per_case
        self.history_size = history_size
        self.history, self.last_mining_epoch = {}, None
        self.set_epoch(0)

    def update(self, records, completed_epoch):
        if set(records) != set(self.case_ids):
            raise ValueError('Mining must cover exactly the training cohort')
        for case_id, nodules in records.items():
            current = self.history.setdefault(case_id, {})
            for nodule, missed in nodules.items():
                key = str(nodule)
                current[key] = (current.get(key, []) + [bool(missed)])[-self.history_size:]
            if set(current) != {str(n) for n in nodules}:
                raise ValueError('Physical GT nodule IDs changed during mining')
        self.last_mining_epoch = int(completed_epoch)

    def case_weight(self, case_id):
        # Once recovered, stop repeating the case unless another GT is missed.
        active = [np.mean(v) for v in self.history.get(case_id, {}).values() if v[-1]]
        return float(max(active, default=0.))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        # Match V4's DistributedSampler exactly before hard repeats begin.
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        base = torch.randperm(len(self.case_ids),generator=generator).tolist()
        weights = np.array([self.case_weight(c) for c in self.case_ids])
        repeats = np.zeros(len(base), dtype=int)
        extra = []
        for _ in range(math.ceil(len(base)*self.fraction)):
            eligible = weights * (repeats < self.max_extra_per_case)
            if eligible.sum() <= 0:
                break
            index = int(rng.choice(len(base), p=eligible/eligible.sum()))
            extra.append(index)
            repeats[index] += 1
        global_indices = base + extra
        padding = (-len(global_indices)) % self.world_size
        if extra:
            global_indices += list(rng.integers(0,len(base),size=padding))
            rng.shuffle(global_indices)
        elif padding:
            global_indices += (base*math.ceil(padding/len(base)))[:padding]
        self.indices = global_indices[self.rank::self.world_size]
        self.summary = dict(epoch=self.epoch+1, ordinary_cases=len(base), extra_cases=len(extra),
            hard_cases=int((weights>0).sum()), ddp_padding=padding,
            global_samples=len(global_indices), local_samples=len(self.indices),
            last_mining_epoch=self.last_mining_epoch,
            repeated_case_ids={self.case_ids[i]:int(n) for i,n in enumerate(repeats) if n},
            schedule_sha256=hashlib.sha256(json.dumps(list(map(int,global_indices))).encode()).hexdigest())

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def state_dict(self):
        return dict(case_ids=self.case_ids, history=self.history, last_mining_epoch=self.last_mining_epoch,
                    seed=self.seed, fraction=self.fraction, max_extra_per_case=self.max_extra_per_case,
                    history_size=self.history_size)

    def load_state_dict(self, state):
        for name in ('case_ids','seed','fraction','max_extra_per_case','history_size'):
            if state[name] != getattr(self,name):
                raise ValueError('Sampler resume mismatch: '+name)
        self.history = state['history']
        self.last_mining_epoch = state['last_mining_epoch']
