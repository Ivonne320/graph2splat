import random
from typing import Any

import numpy
import torch
from torch.utils import data
from collections import defaultdict, deque

from configs import Config
from utils import torch_util
from torch.utils.data import Sampler
from math import ceil

class FairSceneBatchSampler(Sampler):
    """
    - No repeats within an epoch.
    - Prefer at most one sample per scene per round while multiple scenes remain.
    - When some scenes exhaust, keep filling from remaining scenes (may take multiple from one scene).
    - Yields final short batch when drop_last=False.
    Assumes dataset.data_items[i]["scan_id"] exists.
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = False, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._build_epoch_order()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._build_epoch_order()

    def _build_epoch_order(self):
        rng = random.Random(self.seed + self.epoch)
        by_scene = defaultdict(list)
        for idx, it in enumerate(self.dataset.data_items):
            by_scene[it["scan_id"]].append(idx)

        self.scene_ids = list(by_scene.keys())
        rng.shuffle(self.scene_ids)  # new scene order each epoch
        self.scene_queues = {sid: deque(rng.sample(by_scene[sid], k=len(by_scene[sid])))
                             for sid in self.scene_ids}

    def __iter__(self):
        queues = {sid: deque(q) for sid, q in self.scene_queues.items()}
        cycle = deque(self.scene_ids)

        def any_left():
            return any(len(q) > 0 for q in queues.values())

        while any_left():
            batch = []
            while len(batch) < self.batch_size and any_left():
                # who still has items?
                non_empty = [sid for sid in cycle if len(queues[sid]) > 0]
                if not non_empty:
                    break

                if len(non_empty) == 1:
                    # ONLY ONE SCENE LEFT -> take as many as needed from it
                    sid = non_empty[0]
                    take = min(self.batch_size - len(batch), len(queues[sid]))
                    for _ in range(take):
                        batch.append(queues[sid].popleft())
                    # don't rotate; we'll re-check any_left() next
                else:
                    # multiple scenes -> round-robin, at most 1 from current head
                    if len(queues[cycle[0]]) == 0:
                        cycle.rotate(-1)
                        continue
                    sid = cycle[0]
                    batch.append(queues[sid].popleft())
                    cycle.rotate(-1)

            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                yield batch


    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else ceil(n / self.batch_size)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_val_data_loader(
    cfg: Config, dataset: Any
) -> tuple[data.DataLoader, data.DataLoader]:
    _, train_dataloader = get_train_dataloader(cfg, dataset=dataset)
    _, val_dataloader = get_val_dataloader(cfg, dataset=dataset)
    return train_dataloader, val_dataloader


# def get_train_dataloader(
#     cfg: Config, dataset: Any
# ) -> tuple[data.Dataset, data.DataLoader]:
#     train_dataset = dataset(cfg, split="train")
#     train_dataloader = torch_util.build_dataloader(
#         train_dataset,
#         batch_size=cfg.train.batch_size,
#         num_workers=cfg.train.num_workers,
#         shuffle=True,
#         collate_fn=train_dataset.collate_fn,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return train_dataset, train_dataloader

def get_train_dataloader(cfg: Config, dataset: Any) -> tuple[data.Dataset, data.DataLoader]:
    train_dataset = dataset(cfg, split="train")

    use_fair = getattr(cfg.train, "scene_fair_sampling", True)
    if use_fair:
        # NEW: fair, scene-aware batching
        sampler = FairSceneBatchSampler(
            train_dataset,
            batch_size=cfg.train.batch_size,
            drop_last=False,                # <-- allow the final short batch
            seed=getattr(cfg, "seed", 0),
        )
        g = torch.Generator()
        g.manual_seed(getattr(cfg, "seed", 0))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=sampler,          # IMPORTANT: use batch_sampler (not batch_size/shuffle)
            num_workers=cfg.train.num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        # Original behavior (unchanged)
        train_dataloader = torch_util.build_dataloader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )
    return train_dataset, train_dataloader


def get_val_dataloader(
    cfg: Config, dataset: Any
) -> tuple[data.Dataset, data.DataLoader]:
    val_dataset = dataset(cfg, split="train")
    val_dataloader = torch_util.build_dataloader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return val_dataset, val_dataloader


def get_test_dataloader(
    cfg: Config, dataset: Any
) -> tuple[data.Dataset, data.DataLoader]:
    test_dataset = dataset(cfg, split="test")
    test_dataloader = torch_util.build_dataloader(
        test_dataset,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return test_dataset, test_dataloader
