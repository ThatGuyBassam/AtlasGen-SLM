# data/dataset_builder.py
import torch
from torch.utils.data import Dataset, DataLoader
from data.flank_extractor import load_reference, extract_flanks_from_vcf
from tokenizer.kmer_tokenizer import KmerTokenizer, load_vocab

MAX_LENGTH = 256

class GenomicMLMDataset(Dataset):
    def __init__(self, vcf_paths, ref, tokenizer, split="train", max_variants=None):
        self.tokenizer = tokenizer
        self.samples = []

        print(f"Loading {split} split from {len(vcf_paths)} VCF file(s)...")

        for vcf_path in vcf_paths:
            for seq, _ in extract_flanks_from_vcf(
                vcf_path,
                ref,
                split=split,
                max_variants=max_variants
            ):
                token_ids = tokenizer.tokenize(seq)
                token_ids = tokenizer.pad(token_ids, MAX_LENGTH)

                # Defensive Guard 1: Ensure exact sequence length
                if len(token_ids) != MAX_LENGTH:
                    raise ValueError(
                        f"Tokenized sample has length {len(token_ids)}, expected {MAX_LENGTH}."
                    )

                self.samples.append(token_ids)

        # Defensive Guard 2: Prevent cryptic tensor shape crashes on empty splits
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No sequences loaded for split='{split}'. "
                "Check VCF path, reference path, split logic, and max_variants."
            )

        self.samples = torch.tensor(self.samples, dtype=torch.int16)
        print(f"Loaded {len(self.samples)} sequences into {split} dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.samples[idx].long()
        attention_mask = (token_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "mutation_index": 86,
        }

def build_dataloaders(vcf_paths, ref, tokenizer, batch_size=2, max_variants=None):
    train_dataset = GenomicMLMDataset(
        vcf_paths, ref, tokenizer, split="train", max_variants=max_variants
    )
    val_dataset = GenomicMLMDataset(
        vcf_paths, ref, tokenizer, split="val", max_variants=max_variants
    )
    test_dataset = GenomicMLMDataset(
        vcf_paths, ref, tokenizer, split="test", max_variants=max_variants
    )

    # Defensive Guard 3: Only pin memory if a GPU is actually available
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader