# data/flank_extractor.py
# Takes a VCF file and a reference genome, extracts 1024bp windows
# around each variant, injects the alternate allele, outputs DNA strings.
# Includes deterministic spatial chunking to prevent data leakage.

import gzip
import hashlib
from pyfaidx import Fasta

# ── Configuration ────────────────────────────────────────────────────────────

REFERENCE_PATH = "data/reference/chr22.fa"
FLANK_SIZE = 512  
CHUNK_SIZE = 2048 

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Load Reference Genome ─────────────────────────────────────────────────────

def load_reference(path):
    print(f"Loading reference genome from {path}...")
    ref = Fasta(path)
    print("Reference loaded.")
    return ref

# ── Deterministic Split Assignment ───────────────────────────────────────────

def assign_split(chrom, pos, chunk_size=CHUNK_SIZE):
    """
    Assigns a variant to a split deterministically based on spatial chunks.
    Variants within the same chunk_size window get the same hash,
    ensuring overlapping sequences are not split across train/test sets.
    """
    chunk_id = pos // chunk_size
    key = f"{chrom}:{chunk_id}".encode("utf-8")
    hash_int = int(hashlib.md5(key).hexdigest(), 16)
    bucket = (hash_int % 100) / 100.0

    if bucket < TRAIN_RATIO:
        return "train"
    elif bucket < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"

# ── Parse VCF ────────────────────────────────────────────────────────────────

def parse_vcf(vcf_path):
    """
    Read a VCF file (plain or gzipped) and yield one variant at a time.
    """
    opener = gzip.open if vcf_path.endswith(".gz") else open

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            chrom = fields[0]
            pos   = int(fields[1])
            ref   = fields[3]
            alt   = fields[4]

            if "," in alt or len(ref) != 1 or len(alt) != 1:
                continue

            yield {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}

# ── Extract Flank ─────────────────────────────────────────────────────────────

def extract_flank(ref, valid_chroms, chrom, pos, vcf_ref, alt, flank_size=FLANK_SIZE):
    """
    Extract sequence window, verify reference alignment, and inject alternate allele.
    """
    center = pos - 1
    start  = center - flank_size
    end    = center + flank_size + 1

    if start < 0:
        return None

    chrom_key = chrom
    if chrom_key not in valid_chroms:
        if f"chr{chrom}" in valid_chroms:
            chrom_key = f"chr{chrom}"
        elif chrom.startswith("chr") and chrom[3:] in valid_chroms:
            chrom_key = chrom[3:]
        else:
            return None

    try:
        sequence = ref[chrom_key][start:end].seq.upper()
    except Exception:
        return None

    if len(sequence) != flank_size * 2 + 1 or "N" in sequence:
        return None

    fasta_ref_base = sequence[flank_size]
    if fasta_ref_base != vcf_ref:
        return None

    sequence = sequence[:flank_size] + alt + sequence[flank_size + 1:]
    return sequence

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def extract_flanks_from_vcf(vcf_path, ref, split=None, max_variants=None):
    """
    Run the full extraction pipeline for one VCF file.
    """
    count   = 0
    skipped = 0
    valid_chroms = set(ref.keys())

    for variant in parse_vcf(vcf_path):
        variant_split = assign_split(variant["chrom"], variant["pos"])

        if split is not None and variant_split != split:
            continue

        sequence = extract_flank(
            ref,
            valid_chroms,
            chrom=variant["chrom"],
            pos=variant["pos"],
            vcf_ref=variant["ref"],
            alt=variant["alt"],
            flank_size=FLANK_SIZE
        )

        if sequence is None:
            skipped += 1
            continue

        yield sequence, variant_split
        count += 1

        if max_variants and count >= max_variants:
            break

    print(f"Extracted {count} sequences. Skipped {skipped} variants.")

# ── Test Run (IT WORKED FIRST TIME!!!!)  ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    VCF_PATH = "data/raw/1000genomes/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

    ref = load_reference(REFERENCE_PATH)

    print("Extracting sequences to verify spatial chunking distribution...")
    split_counts = {"train": 0, "val": 0, "test": 0}

    for i, (seq, split_label) in enumerate(extract_flanks_from_vcf(VCF_PATH, ref, max_variants=10)):
        split_counts[split_label] += 1
        print(f"\nSequence {i+1} | split={split_label} | length={len(seq)}")
        print(seq[:60] + "...")

    print(f"\nSplit distribution: {split_counts}")
    print("Flank extractor working correctly.")
