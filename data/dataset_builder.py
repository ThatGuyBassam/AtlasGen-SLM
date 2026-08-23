# data/dataset_builder.py
# Loads tokenized sequences from the disk shards written by
# data/shard_builder.py. This is what the trainer actually pulls batches
# from — it does NOT re-run flank extraction or tokenization at train time.

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class ShardedGenomicMLMDataset(Dataset):
    """
    Lazily loads one shard file at a time from disk instead of holding the
    entire split in memory.

    Each split (train/val/test) is a directory of shard_XXXXX.pt files,
    each an int16 tensor of shape [N, max_length], plus a manifest.json
    recording shard count and total sequence count.

    Caching: this Dataset keeps only ONE shard resident in memory at a
    time. That only stays cheap if consecutive __getitem__ calls tend to
    hit the same shard — which is not true under plain random shuffling
    (see ShardShuffleSampler below, which is what makes that assumption
    hold in practice).
    """

    def __init__(self, shard_dir, split):
        self.split = split
        self.shard_dir = Path(shard_dir) / split

        manifest_path = self.shard_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest found at {manifest_path}. "
                f"Run data/shard_builder.py for split='{split}' first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.num_shards = manifest["num_shards"]
        self.total_sequences = manifest["total_sequences"]
        self.shard_size = manifest["shard_size"]
        self.max_length = manifest["max_length"]

        self._cached_shard_idx = None
        self._cached_shard_tensor = None

    def __len__(self):
        return self.total_sequences

    def _load_shard(self, shard_idx):
        if self._cached_shard_idx == shard_idx:
            return self._cached_shard_tensor

        shard_path = self.shard_dir / f"shard_{shard_idx:05d}.pt"
        tensor = torch.load(shard_path, weights_only=True)

        self._cached_shard_idx = shard_idx
        self._cached_shard_tensor = tensor
        return tensor

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(idx)

        shard_idx = idx // self.shard_size
        offset = idx % self.shard_size

        shard_tensor = self._load_shard(shard_idx)
        token_ids = shard_tensor[offset].long()

        # pad_id is always vocab index 0 by construction (see kmer_tokenizer.build_vocab).
        attention_mask = (token_ids != 0).long()

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "mutation_index": 86,
            "index": idx,  # stable per-sample id, used to seed MLM masking deterministically
        }


class ShardShuffleSampler(Sampler):
    """
    Shuffles shard order AND within-shard order every epoch, while keeping
    all accesses to one shard contiguous in time.

    Why not just DataLoader(shuffle=True): with a fully random global
    shuffle, consecutive samples land in essentially random shards, so the
    Dataset's single-shard cache would miss on almost every __getitem__ —
    reloading a shard file from disk per sample. This sampler still gives
    a genuinely shuffled training order (both across shards and within
    each shard), it just does the shuffling at a granularity that respects
    the on-disk layout.
    """

    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        # Call this once per epoch from the training loop so shard/within-
        # shard order actually changes across epochs instead of repeating.
        self.epoch = epoch

    def __iter__(self):
        shard_size = self.dataset.shard_size
        num_shards = self.dataset.num_shards
        total = self.dataset.total_sequences

        rng = random.Random(self.seed + self.epoch)

        shard_order = list(range(num_shards))
        if self.shuffle:
            rng.shuffle(shard_order)

        for shard_idx in shard_order:
            start = shard_idx * shard_size
            end = min(start + shard_size, total)
            local_indices = list(range(start, end))
            if self.shuffle:
                rng.shuffle(local_indices)
            yield from local_indices

    def __len__(self):
        return self.dataset.total_sequences


def build_dataloaders(
    shard_dir,
    batch_size=32,
    num_workers=2,
    seed=0,
    train_collate_fn=None,
    eval_collate_fn=None,
):
    """
    train_collate_fn / eval_collate_fn: optional callables passed straight
    through to DataLoader(collate_fn=...). Because DataLoader runs
    collate_fn inside worker processes (when num_workers > 0), this is
    where MLM masking should be plugged in — see train/trainer.py's
    MLMCollator — rather than masking after batches reach the main
    process/training loop.
    """
    train_dataset = ShardedGenomicMLMDataset(shard_dir, split="train")
    val_dataset = ShardedGenomicMLMDataset(shard_dir, split="val")
    test_dataset = ShardedGenomicMLMDataset(shard_dir, split="test")

    pin_memory = torch.cuda.is_available()

    train_sampler = ShardShuffleSampler(train_dataset, shuffle=True, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=train_collate_fn,
    )

    # Val/test don't need shuffling at all — deterministic order is fine,
    # and a plain sequential sampler naturally stays shard-local too.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=eval_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=eval_collate_fn,
    )

    return train_loader, val_loader, test_loader, train_sampler
