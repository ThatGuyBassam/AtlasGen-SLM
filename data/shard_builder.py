# data/shard_builder.py
# Reads VCF(s) + reference, extracts flanks, tokenizes, and writes the
# result to disk as fixed-size shards. This is the offline preprocessing
# step — run once per phase/dataset, before training ever starts.
#
# Why sharded instead of one giant in-memory tensor (the old approach):
# - Phase 1c+ regional extracts and later phases pull in far more sequences
#   than comfortably fit in RAM at once on this hardware.
# - Shards let the Dataset load one file at a time instead of the whole split.
# - Writing is atomic (temp file + os.replace) so a crash or Ctrl+C mid-write
#   never leaves a corrupted shard that silently poisons training.

import os
import json
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")

from data.flank_extractor import load_reference, extract_flanks_from_vcf
from tokenizer.kmer_tokenizer import KmerTokenizer, load_vocab

MAX_LENGTH = 256
SHARD_SIZE = 50_000  # sequences per shard file


def check_vcfs_exist(vcf_paths):
    """
    Hard error on any missing VCF. Silently skipping a missing file would
    mean a phase trains on less data than intended without anyone noticing
    until val loss looks off — fail loud and immediately instead.
    """
    missing = [p for p in vcf_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "The following VCF file(s) are missing on disk:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nShard building requires every listed VCF to be present. Aborting."
        )


def atomic_save(tensor, path):
    """
    Write a tensor to disk atomically.

    torch.save() first to a temp file in the same directory, then
    os.replace() it into the real filename. os.replace is atomic on both
    POSIX and Windows: a reader will only ever see either the old file (if
    any) or the fully-written new one, never a half-written one from an
    interrupted save.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, path)


def build_shards_single_pass(
    vcf_paths,
    ref,
    tokenizer,
    output_dir,
    max_variants_per_split=None,
    shard_size=SHARD_SIZE,
):
    """
    Reads every VCF file exactly ONCE and routes each variant to the
    correct split's buffer/shard as it's read, instead of re-reading the
    whole file separately for train, then again for val, then again for
    test. On multi-hundred-MB VCFs, that's the difference between one
    pass and three over the same data.

    extract_flanks_from_vcf(..., split=None) already yields every
    successfully-extracted variant along with its assigned split label
    without filtering — this function is what actually uses that mode.
    """
    check_vcfs_exist(vcf_paths)

    splits = ["train", "val", "test"]
    split_dirs = {}
    buffers = {s: [] for s in splits}
    shard_indices = {s: 0 for s in splits}
    total_written = {s: 0 for s in splits}

    for s in splits:
        d = Path(output_dir) / s
        d.mkdir(parents=True, exist_ok=True)
        split_dirs[s] = d

    def flush(s):
        if not buffers[s]:
            return
        tensor = torch.tensor(buffers[s], dtype=torch.int16)
        shard_path = split_dirs[s] / f"shard_{shard_indices[s]:05d}.pt"
        atomic_save(tensor, shard_path)
        total_written[s] += len(buffers[s])
        print(f"  [{s}] wrote {shard_path.name} ({len(buffers[s])} sequences)")
        buffers[s] = []
        shard_indices[s] += 1

    def split_is_full(s):
        if max_variants_per_split is None:
            return False
        return (total_written[s] + len(buffers[s])) >= max_variants_per_split

    def all_splits_full():
        return all(split_is_full(s) for s in splits)

    print(f"Building shards for all splits in a single pass -> {output_dir}")

    for vcf_path in vcf_paths:
        if all_splits_full():
            break

        print(f"  Reading {vcf_path} ...")
        stats = {"extracted": 0, "skipped": 0}

        for seq, split_label in extract_flanks_from_vcf(vcf_path, ref, split=None, stats=stats):
            if split_is_full(split_label):
                # This split already has enough — skip tokenizing/storing
                # it, but keep reading the file for the OTHER splits.
                if all_splits_full():
                    break
                continue

            token_ids = tokenizer.tokenize(seq)
            token_ids = tokenizer.pad(token_ids, MAX_LENGTH)

            if len(token_ids) != MAX_LENGTH:
                raise ValueError(
                    f"Tokenized sample has length {len(token_ids)}, "
                    f"expected {MAX_LENGTH}."
                )

            buffers[split_label].append(token_ids)

            if len(buffers[split_label]) >= shard_size:
                flush(split_label)

        # Read from `stats`, not from the generator's own trailing print --
        # that print gets skipped if we broke out of the loop above before
        # the generator was naturally exhausted (see extract_flanks_from_vcf
        # docstring). `stats` stays accurate either way since it's updated
        # synchronously before each yield/skip.
        print(f"    -> extracted {stats['extracted']}, skipped {stats['skipped']} (this file)")

    for s in splits:
        flush(s)

    manifests = {}
    for s in splits:
        if total_written[s] == 0:
            raise RuntimeError(
                f"No sequences written for split='{s}'. "
                "Check VCF path, reference path, and split logic."
            )

        manifest = {
            "split": s,
            "num_shards": shard_indices[s],
            "total_sequences": total_written[s],
            "shard_size": shard_size,
            "max_length": MAX_LENGTH,
        }

        manifest_path = split_dirs[s] / "manifest.json"
        tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
        with open(tmp_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest_path, manifest_path)

        print(f"Finished '{s}': {total_written[s]} sequences across {shard_indices[s]} shard(s).")
        manifests[s] = manifest

    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build disk shards for AtlasGen-SLM training.")
    parser.add_argument("--vcf", nargs="+", required=True, help="One or more VCF file paths.")
    parser.add_argument("--reference", default="data/reference/chr22.fa")
    parser.add_argument("--vocab", default="tokenizer/vocab.json")
    parser.add_argument("--output-dir", default="data/shards")
    parser.add_argument("--max-variants", type=int, default=None,
                         help="Cap variants PER SPLIT (not total), useful for smoke tests.")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    ref = load_reference(args.reference)
    vocab = load_vocab(args.vocab)
    tokenizer = KmerTokenizer(vocab=vocab)

    build_shards_single_pass(
        vcf_paths=args.vcf,
        ref=ref,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_variants_per_split=args.max_variants,
        shard_size=args.shard_size,
    )
