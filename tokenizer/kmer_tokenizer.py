# tokenizer/kmer_tokenizer.py
# Converts raw DNA sequences into non-overlapping 6-mer token IDs.
# Stride = K (no overlap) eliminates MLM information leakage.
# UNK granularity: [UNK_1] through [UNK_6] based on N count in k-mer.
# Mutation center invariant: always at position 2 of token index 85.

import json
import itertools
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

K = 6
BASES = ["A", "C", "G", "T"]

# Standard special tokens
PAD_TOKEN  = "[PAD]"
MASK_TOKEN = "[MASK]"
CLS_TOKEN  = "[CLS]"
SEP_TOKEN  = "[SEP]"
UNK_TOKEN  = "[UNK]"  # fallback only — should never appear in practice

# Granular N-uncertainty tokens
# [UNK_N] means exactly N bases in this 6-mer are unknown
UNK_TOKENS = [f"[UNK_{i}]" for i in range(1, K + 1)]  # [UNK_1] ... [UNK_6]

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN] + UNK_TOKENS

# Mutation center invariant — fixed constants for all samples
# With FLANK_SIZE=512: center base is always at sequence position 512
# 512 // 6 = 85 remainder 2 → always token index 85, position 2 within token
MUTATION_TOKEN_INDEX = 85   # which token contains the mutation (after CLS offset = 86)
MUTATION_INTRA_POS   = 2    # which position within that token (0-5)

# ── Vocabulary Builder ────────────────────────────────────────────────────────

def build_vocab():
    """
    4^6 = 4,096 k-mers
    + 5 standard special tokens
    + 6 UNK granularity tokens
    = 4,107 total vocabulary size.
    """
    vocab = {}

    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab[token] = len(vocab)

    for kmer in itertools.product(BASES, repeat=K):
        kmer_str = "".join(kmer)
        if kmer_str not in vocab:
            vocab[kmer_str] = len(vocab)

    return vocab

def save_vocab(vocab, path="tokenizer/vocab.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {path} ({len(vocab)} tokens)")

def load_vocab(path="tokenizer/vocab.json"):
    with open(path, "r") as f:
        vocab = json.load(f)
    print(f"Vocabulary loaded from {path} ({len(vocab)} tokens)")
    return vocab

# ── Tokenizer ─────────────────────────────────────────────────────────────────

class KmerTokenizer:
    """
    Non-overlapping stride-6 k-mer tokenizer with granular N-uncertainty tokens.

    Key invariant:
        The mutation is always at sequence position 512 (FLANK_SIZE=512).
        512 // 6 = token index 85, intra-token position 2.
        This is constant across ALL training samples — no positional noise.

    UNK granularity:
        Instead of one [UNK] for all N-containing k-mers, we use [UNK_1]..[UNK_6]
        where the number reflects how many bases are unknown.
        ANNNGT → [UNK_4] (4 unknowns)
        NNNNNN → [UNK_6] (6 unknowns)
        This preserves uncertainty resolution rather than collapsing it.
    """

    def __init__(self, vocab=None, vocab_path="tokenizer/vocab.json", k=K):
        self.k = k
        self.vocab = vocab if vocab is not None else load_vocab(vocab_path)
        self.id_to_token = {v: key for key, v in self.vocab.items()}

        self.pad_id  = self.vocab[PAD_TOKEN]
        self.unk_id  = self.vocab[UNK_TOKEN]
        self.mask_id = self.vocab[MASK_TOKEN]
        self.cls_id  = self.vocab[CLS_TOKEN]
        self.sep_id  = self.vocab[SEP_TOKEN]

        # Precompute UNK_N IDs for fast lookup during tokenization
        self.unk_n_ids = {
            i: self.vocab[f"[UNK_{i}]"] for i in range(1, self.k + 1)
        }

    def tokenize(self, sequence):
        """
        Tokenize a DNA sequence using non-overlapping stride-K k-mers.

        - Pads sequence end with N so length is a multiple of K (no bases dropped)
        - N-containing k-mers → [UNK_N] where N = number of N bases
        - CLS prepended, SEP appended
        """
        token_ids = [self.cls_id]

        remainder = len(sequence) % self.k
        if remainder != 0:
            sequence += "N" * (self.k - remainder)

        for i in range(0, len(sequence), self.k):
            kmer = sequence[i:i + self.k]
            if len(kmer) < self.k:
                break

            n_count = kmer.count("N")
            if n_count > 0:
                token_ids.append(self.unk_n_ids[n_count])
            else:
                token_ids.append(self.vocab.get(kmer, self.unk_id))

        token_ids.append(self.sep_id)
        return token_ids

    def decode(self, token_ids):
        return [self.id_to_token.get(tid, UNK_TOKEN) for tid in token_ids]

    def pad(self, token_ids, max_length):
        """Pad or truncate to max_length, always preserving SEP at end."""
        n = len(token_ids)
        if n == max_length:
            return token_ids
        if n > max_length:
            return token_ids[:max_length - 1] + [self.sep_id]
        return token_ids + [self.pad_id] * (max_length - n)

    def get_mutation_token_position(self):
        """
        Returns the index in the token list where the mutation sits.
        +1 accounts for the CLS token prepended at position 0.
        This is a fixed constant — same for every sample.
        """
        return MUTATION_TOKEN_INDEX + 1  # = 86

    @property
    def vocab_size(self):
        return len(self.vocab)

# ── Test Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.flank_extractor import load_reference, extract_flanks_from_vcf

    print("Building vocabulary...")
    vocab = build_vocab()
    save_vocab(vocab)
    print(f"Vocab size: {len(vocab)}")
    print(f"UNK tokens: { {k: vocab[k] for k in UNK_TOKENS} }")

    tokenizer = KmerTokenizer(vocab=vocab)
    print(f"\nMutation always at token index: {tokenizer.get_mutation_token_position()}")
    print(f"Mutation intra-token position: {MUTATION_INTRA_POS} (of 0-5)")

    print("\nLoading reference and extracting one sequence...")
    ref = load_reference("data/reference/chr22.fa")
    VCF = "data/raw/1000genomes/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

    for seq, split in extract_flanks_from_vcf(VCF, ref, max_variants=1):
        print(f"\nSequence length: {len(seq)}")
        print(f"First 60 bases: {seq[:60]}...")

        token_ids = tokenizer.tokenize(seq)
        print(f"\nTotal tokens: {len(token_ids)}")
        print(f"First 10 token IDs: {token_ids[:10]}")
        print(f"Last 10 token IDs:  {token_ids[-10:]}")
        print(f"Decoded first 10: {tokenizer.decode(token_ids[:10])}")

        # Verify mutation token
        mut_pos = tokenizer.get_mutation_token_position()
        print(f"\nToken at mutation position ({mut_pos}): "
              f"{token_ids[mut_pos]} → {tokenizer.decode([token_ids[mut_pos]])}")

        padded = tokenizer.pad(token_ids, max_length=256)
        print(f"\nPadded to 256: {len(padded)} tokens")
        print(f"SEP preserved: {padded[-1] == tokenizer.sep_id}")

    print("\nTokenizer working correctly.")
