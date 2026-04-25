# AtlasGen-SLM — Technical Architecture

## Overview

AtlasGen-SLM is an encoder-only transformer trained to classify the pathogenicity of human genomic variants. It is trained in four sequential phases, moving from global genomic diversity down to Moroccan-specific sequences and finally to supervised pathogenicity classification.

The model runs entirely offline on consumer-grade hardware after training. Inference requires no internet connection, no cloud compute, and no external API calls.

---

## The Core Pipeline

```
Input: genomic variant (chr, pos, ref, alt)
          ↓
[1] Flank Extraction
    Query GRCh38 reference genome
    Extract 1024bp window centered on variant position
    Inject alternate allele at center
    Output: raw DNA string
          ↓
[2] K-mer Tokenization
    Slide window of k=6 across the DNA string
    Convert each 6-mer to an integer token ID
    Vocabulary size: 4^6 = 4,096 tokens
    Output: sequence of integer token IDs
          ↓
[3] Transformer Encoder
    Token embeddings + positional encodings
    Multi-head self-attention layers
    Feed-forward layers
    Output: contextual sequence representation
          ↓
[4] Classification Head (Phase 4 only)
    Linear projection onto 3 classes
    Output: Benign / Pathogenic / Uncertain
    + confidence score per class
```

---

## Why K-mer Tokenization

DNA has no natural word boundaries — no spaces, no punctuation, no syntactic delimiters. K-mers solve this by treating every overlapping subsequence of length k as a token.

With k=6:
```
Sequence:  A T C G G A T C C A ...
6-mers:    ATCGGA
            TCGGAT
             CGGATC
              GGATCC ...
```

This captures local sequence context while keeping the vocabulary compact enough for a 15M parameter model to learn effectively. DNABERT, one of the foundational genomic language models, established k=6 as the standard for this scale.

**Handling ambiguous bases:** Genomic sequences regularly contain "N" bases — positions where the nucleotide is unknown or unsequenced. Any 6-mer containing an N is skipped entirely during tokenization rather than assigned an `<UNK>` token. N bases occur in low-quality sequencing regions; the model should not learn anything from them. Skipping is cleaner than representing uncertainty about uncertainty.

---

## Why VCF-Guided Flank Extraction

Training directly on raw whole genome sequences is computationally impossible at 8GB VRAM — a full human genome is 3.2 billion base pairs. Transformer attention scales quadratically O(L²) with sequence length, making direct WGS ingestion infeasible within that memory envelope.

VCF-guided flank extraction solves this:

1. Parse the VCF to get variant coordinates
2. Extract a fixed 1024bp window from GRCh38 around each variant
3. Inject the alternate allele at the variant position
4. The model sees the variant in its natural genomic context

This compresses the training signal to only the informative positions while preserving the sequence grammar the model needs to learn. Every training example is a meaningful, self-contained sequence chunk — not a truncated fragment of a larger genome.

---

## Model Architecture

### Specifications

| Component | Value |
|---|---|
| Architecture | Encoder-only Transformer |
| Parameters | ~15 million |
| Layers | 12 |
| Attention heads | 8 |
| Hidden dimension | 512 |
| Feed-forward dimension | 2048 |
| Max sequence length | 1024 tokens |
| Vocabulary size | 4,096 (6-mers) + special tokens |

### Components

**Token Embedding Layer**
Maps each integer token ID to a 512-dimensional vector. The model learns what each 6-mer means in the context of genomic sequences.

**Positional Encoding**
Sinusoidal positional encodings added to token embeddings. Gives the model information about where in the 1024bp window each token sits — critical for understanding regulatory distance effects.

**Transformer Encoder Blocks (×12)**
Each block contains:
- Multi-head self-attention (8 heads)
- Layer normalization
- Feed-forward network (512 → 2048 → 512)
- Residual connections

**Classification Head (Phase 4)**
A linear layer projecting the [CLS] token representation to 3 output classes: Benign, Pathogenic, Uncertain.

---

## Four-Phase Training

### Phase 1 — Global Pretraining
**Objective:** Learn the fundamental grammar of human DNA across global diversity.

The model trains from scratch on sequences derived from globally diverse VCFs. For every variant in the 1000 Genomes Project, SGDP, HGDP, gnomAD, and TOPMed datasets, the pipeline extracts the 1024bp flank, tokenizes it, and trains the model using masked language modeling (MLM) — randomly masking 15% of tokens and training the model to predict them. This is the same self-supervised objective used in BERT.

**Why MLM:** It forces the model to learn sequence context without requiring any labels. The model learns that certain k-mers tend to appear together, that certain patterns signal splice sites, that certain motifs are conserved — all from raw sequence alone.

### Phase 2 — African & Maghrebi Fine-tuning
**Objective:** Shift the model's learned representations toward African and Maghrebi genomic architecture.

Continue MLM training on sequences from H3Africa, EGP1K, GME Variome, and Saudi Genome Program data. The model updates its weights to better represent the linkage disequilibrium patterns, runs of homozygosity, and rare variant distributions specific to these populations.

### Phase 3 — Moroccan-Specific Fine-tuning
**Objective:** Specialize on the specific haplotype structures and allele frequencies of the Moroccan genome.

Fine-tune on the Moroccan Genome Project (EGAD50000000750), Moroccan urban array data, and Algerian Amazigh sequences. At this stage the model has a strong prior on African genomic variation and uses Phase 3 data to anchor on the Moroccan-specific signal.

### Phase 4 — Pathogenic Variant Classification
**Objective:** Convert the sequence encoder into a supervised pathogenicity classifier.

Add the classification head and switch from MLM to supervised training on labeled variants from ClinVar, HGMD, LOVD, and Mastermind.

**Curriculum learning strategy:**
1. Train on the union of ClinVar + HGMD to maximize variant diversity exposure
2. Apply penalty weighting: where HGMD labels a variant pathogenic but ClinVar labels it benign, train the model to output benign — using ClinVar as the high-confidence regulatory layer over HGMD's noisier historical data
3. Fine-tune on Moroccan-specific pathogenic variants from MGDD and published case studies

---

## Hardware & Memory Management

### Target Hardware
NVIDIA RTX 4060 — 8GB VRAM

### Memory Strategy

**FlashAttention**
Standard attention materializes the full N×N attention matrix in GPU memory — at sequence length 1024 this is prohibitive. FlashAttention computes attention in blocks without materializing the full matrix, reducing memory footprint from O(L²) to O(L) while producing mathematically identical results.

**PagedAdamW**
The AdamW optimizer stores momentum and variance tensors for every parameter — effectively tripling the memory cost of the model weights. PagedAdamW offloads optimizer states to CPU memory when GPU memory is under pressure, recovering VRAM for activations and batch data.

**fp16 Precision**
All tensors stored and computed in 16-bit floating point. Halves memory usage relative to fp32 with negligible impact on model quality at this parameter scale.

**Gradient Checkpointing**
During backpropagation, intermediate activations are recomputed on-demand rather than stored in memory. Trades compute time for VRAM — a necessary tradeoff at 8GB.

---

## Validation

Validation is anchored to monogenic disease variants with known, well-characterized population-specific profiles:

**HBB (beta-globin gene)**
Sickle cell disease and beta-thalassemia variants. North African populations carry distinct HBB haplotypes at elevated frequencies that Western databases systematically misclassify. The model's performance on HBB variants provides a direct measure of its ability to contextualize North African-common pathogenic variants.

**G6PD (glucose-6-phosphate dehydrogenase)**
G6PD deficiency variants are prevalent across the Mediterranean and North Africa. Several variants benign in heterozygous North African females are flagged as pathogenic by Western tools. Performance on G6PD tests the model's ability to correctly handle population-frequency-dependent pathogenicity.

**Primary metric:** VUS reclassification rate — the proportion of variants classified as uncertain by Western reference tools that AtlasGen-SLM resolves to a confident benign or pathogenic call.

---

## Inference (Post-Training)

Once trained, the model weights are saved as a `.pt` file. Inference is:

```python
model = AtlasGenSLM.load("weights/atlasgen_phase4.pt")
result = model.classify(chrom="11", pos=5246696, ref="A", alt="T")
# → {"label": "Pathogenic", "confidence": 0.94}
```

No internet. No API. No cloud. The weights file is the entire model.
