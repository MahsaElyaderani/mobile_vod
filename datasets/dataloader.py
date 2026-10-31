from __future__ import annotations

from typing import List, Sequence, Tuple
from functools import partial
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from .paired_frames import stack_frames


def configure_worker_sharing(worker_id, strategy):
    # Spawned workers do not inherit the parent's Python module settings.
    torch.multiprocessing.set_sharing_strategy(strategy)


def video_collate_fn(batch: Sequence[Tuple[Tensor, list[dict]]]):
    """Collate fixed-length video clips while preserving ragged GT boxes."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    frames, targets = zip(*batch)
    first_shape = tuple(frames[0].shape)
    if any(tuple(x.shape) != first_shape for x in frames):
        raise ValueError("All clips in a batch must have the same [T,C,H,W] shape.")
    return stack_frames(frames, dim=0), list(targets)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: int = 1,
    sharing_strategy: str | None = None,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive.")
    loader_options = {}
    if sharing_strategy is not None:
        if sharing_strategy not in torch.multiprocessing.get_all_sharing_strategies():
            raise ValueError(f"Unsupported tensor sharing strategy: {sharing_strategy}")
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
        if sharing_strategy is not None:
            loader_options["worker_init_fn"] = partial(configure_worker_sharing, strategy=sharing_strategy)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=video_collate_fn,
        persistent_workers=(num_workers > 0),
        **loader_options,
    )


class SeededSamples(Dataset):
    """Opt-in reproducible augmentation, independent of worker count/prefetch."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                seed = (self.seed * len(self) + index) % (2**63 - 1)
                random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                return self.dataset[index]
        finally:
            random.setstate(state)


def budget_loader(dataset, args, epoch, start_batch, batches):
    """Rebuild only the unconsumed slice of a deterministic epoch permutation."""
    from .classification import classification_collate
    seed = args.seed + epoch
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator)
    start = start_batch * args.batch_size
    indices = order[start:start + batches * args.batch_size].tolist()
    options = {}
    if args.num_workers:
        options['prefetch_factor'] = args.prefetch_factor
        if args.sharing_strategy:
            options['worker_init_fn'] = partial(configure_worker_sharing, strategy=args.sharing_strategy)
    if args.sharing_strategy:
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    return DataLoader(SeededSamples(dataset, seed), sampler=indices, batch_size=args.batch_size,
                      num_workers=args.num_workers, drop_last=True, generator=generator,
                      pin_memory=torch.device(args.device).type == 'cuda',
                      collate_fn=classification_collate if args.task == 'classification' else video_collate_fn,
                      **options)
