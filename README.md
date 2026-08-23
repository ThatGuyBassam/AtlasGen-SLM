# AtlasGen-SLM

> **A student-built proof of concept exploring small language models for genomic representation learning, with a focus on North African and Moroccan populations.**

---

## ⚠️ Project Status & Disclaimer

**AtlasGen-SLM is an early-stage proof-of-concept and student research project.**

This project is being developed independently as an exploration of whether a relatively small genomic language model can learn useful representations of human genomic sequence and potentially provide a foundation for downstream variant interpretation.

**Training has not yet started.** The current repository contains the implemented model architecture, tokenizer, genomic preprocessing pipeline, dataset/sharding infrastructure, and training code intended for the initial experiments.

AtlasGen-SLM is **not a clinical tool, diagnostic system, validated medical device, or production-ready AI model**.

No claims are currently being made about clinical accuracy, pathogenicity prediction, VUS resolution, or superiority over existing tools.

The project's purpose at this stage is simply to build and test the idea.

---

## What is AtlasGen-SLM?

AtlasGen-SLM is an experimental **small genomic language model** designed to learn patterns in human genomic sequence.

The project explores a simple question:

> **Can a relatively small Transformer, trained on genomic sequence and variation, learn useful representations that could eventually help with variant interpretation in populations that are underrepresented in existing genomic datasets?**

The particular motivation is North African and Moroccan genomic representation.

The project is **not intended to replace established clinical interpretation pipelines**. Instead, it is a student-led proof of concept intended to explore the technical feasibility of the approach.

---

## The Idea

The initial model is an **encoder-only Transformer** trained using masked language modeling.

The basic concept is:

```text
Human genomic sequence
        │
        ▼
1024 bp genomic window
        │
        ▼
6-mer tokenization
        │
        ▼
AtlasGen-SLM
        │
        ▼
Learned genomic representation
        │
        ▼
Potential downstream applications
```

The eventual research direction is to investigate whether those learned representations can be useful for comparing reference and alternate genomic sequences.

For example:

```text
             Variant
                │
        ┌───────┴───────┐
        ▼               ▼
   REF sequence     ALT sequence
        │               │
        ▼               ▼
   AtlasGen-SLM     AtlasGen-SLM
        │               │
        ▼               ▼
   REF embedding    ALT embedding
        │               │
        └───────┬───────┘
                ▼
       Representation comparison
                │
                ▼
        Future classifier
```

**This downstream classifier does not currently exist as a validated system.**

It is one of the eventual experiments the project is intended to investigate.

---

## Why North African Genomics?

Many genomic datasets and computational genomics tools have historically had stronger representation of European populations than many other populations worldwide.

North African populations are particularly interesting from a population-genomics perspective because of their complex demographic history and mixture of ancestral influences.

For a student project based in Morocco, this provides an interesting research question:

> **Could population-aware genomic pretraining eventually produce representations that are useful for variants encountered in North African populations?**

AtlasGen-SLM is an attempt to explore that question.

It is **not assumed that the answer will be yes**.

That is precisely what the proof of concept is intended to test.

---

## Current Model

The current architecture is a roughly **43 million parameter encoder-only Transformer**.

| Component | Configuration |
|---|---|
| Architecture | Encoder-only Transformer |
| Parameters | ~43M |
| Hidden dimension | 512 |
| Transformer layers | 12 |
| Attention heads | 8 |
| Feed-forward | SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Attention | PyTorch SDPA |
| Vocabulary | 4,107 tokens |
| Genomic token | 6-mer |
| Training objective | Masked language modeling |
| Intended precision | BF16 |

The architecture is deliberately relatively small.

One of the goals of the project is to investigate what can be achieved with a model that can realistically be experimented with on **consumer hardware**, rather than requiring a large multi-GPU research cluster.

---

## Tokenization

AtlasGen-SLM uses a custom **6-mer tokenizer**.

The vocabulary consists of:

```text
4,096 possible DNA 6-mers
+
11 special / unknown tokens
=
4,107 tokens
```

The tokenizer converts genomic DNA into non-overlapping 6-mer tokens.

The implementation can be found in:

```text
tokenizer/
├── kmer_tokenizer.py
└── vocab.json
```

---

## Genomic Input

The preprocessing pipeline is designed around variant-centered genomic sequence.

The current target representation is:

```text
1024 bp genomic window
        │
        ▼
6-mer tokenization
        │
        ▼
256 genomic tokens
```

The pipeline extracts sequence from a reference genome around genomic variants and prepares the resulting sequences for model training.

---

## Data Pipeline

The project includes infrastructure for processing genomic variant datasets into training-ready shards.

```text
VCF / genomic dataset
        │
        ▼
Variant filtering
        │
        ▼
Deterministic splitting
        │
        ▼
Reference sequence extraction
        │
        ▼
Variant-centered sequence
        │
        ▼
6-mer tokenization
        │
        ▼
Disk shards
        │
        ▼
Lazy dataset loading
        │
        ▼
Model training
```

The relevant components are:

```text
data/
├── flank_extractor.py
├── shard_builder.py
└── dataset_builder.py
```

The sharded design is intended to allow large genomic datasets to be processed without requiring the entire dataset to fit into system memory.

---

## Planned Training

**Training has not started yet.**

The current implementation is the infrastructure for the first training experiments.

The initial training objective is masked language modeling:

```text
Original sequence:

ATGCGT CAGTAC GGTCCA ...

Masked sequence:

ATGCGT [MASK] GGTCCA ...

                    │
                    ▼

               Transformer

                    │
                    ▼

             Predicted token
```

The first experiments will determine whether the model can successfully learn genomic sequence representations at all.

Only after that will it make sense to investigate whether those representations are useful for downstream variant-related tasks.

---

## Planned Dataset Strategy

The project is organized into several planned stages.

### Phase 1a — Global Foundation

The first stage is intended to expose the model to broad human genomic variation.

The purpose is to learn general genomic representations rather than directly predict pathogenicity.

### Phase 1b — Genome Breadth

The second stage is intended to expand genomic coverage and expose the model to a broader distribution of human genomic variation.

### Phase 1c — North African & Clinical-Locus Enrichment

A later stage is planned to increase representation of North African populations and selected clinically relevant genomic regions.

Potential resources include:

- 1000 Genomes
- HGDP
- SGDP
- gnomAD
- additional population-specific resources

Selected clinical regions are also being considered for targeted enrichment.

**These phases describe the planned experimental roadmap. They should not be interpreted as completed training runs.**

---

## Planned Downstream Experiment

If foundation-model pretraining is successful, a later stage will investigate whether the model can be adapted for variant-effect prediction.

A possible experiment would compare:

```text
Reference sequence
        │
        ▼
     Encoder
        │
        ▼
   REF embedding

        versus

Alternate sequence
        │
        ▼
     Encoder
        │
        ▼
   ALT embedding
```

The resulting representations could then be used by a small downstream classifier.

Potential classes could include benign, pathogenic, and uncertain categories, depending on the eventual dataset and experimental design.

However, **this is a future research direction, not a current capability of AtlasGen-SLM**.

---

## Evaluation

The first evaluation question is simply whether the model can learn useful genomic representations.

Initial experiments are expected to examine metrics such as:

- masked-token loss
- perplexity
- token prediction accuracy
- representation quality

If downstream variant classification is eventually implemented, additional metrics could include:

- AUROC
- AUPRC
- sensitivity
- specificity
- F1
- calibration

Population-stratified evaluation would also be important given the project's motivation.

---

## Hardware

AtlasGen-SLM is being developed on relatively accessible hardware:

```text
GPU:
NVIDIA RTX 4060 — 8 GB VRAM

CPU:
AMD Ryzen 7 7700
```

The project deliberately operates under relatively tight computational constraints.

This is part of the experiment itself:

> **How much can a small student-built genomic language model accomplish without access to a large research cluster?**

---

## Project Structure

```text
AtlasGen-SLM/
│
├── model/
│   └── transformer.py
│
├── tokenizer/
│   ├── kmer_tokenizer.py
│   └── vocab.json
│
├── data/
│   ├── flank_extractor.py
│   ├── shard_builder.py
│   └── dataset_builder.py
│
├── train/
│   └── trainer.py
│
├── eval/
│
├── docs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Large genomic datasets, reference genomes, generated shards, and model checkpoints are intentionally not stored in the repository.

---

## Project Status

| Component | Status |
|---|---|
| Project concept | ✅ Defined |
| Repository | ✅ Active |
| 6-mer tokenizer | ✅ Implemented |
| 4,107-token vocabulary | ✅ Implemented |
| Genomic flank extraction | ✅ Implemented |
| Dataset splitting | ✅ Implemented |
| Dataset sharding | ✅ Implemented |
| Lazy dataset loading | ✅ Implemented |
| 43M Transformer | ✅ Implemented |
| RoPE | ✅ Implemented |
| SDPA | ✅ Implemented |
| RMSNorm | ✅ Implemented |
| SwiGLU | ✅ Implemented |
| Training pipeline | ✅ Implemented |
| Model training | ⏳ **Not started** |
| Phase 1a | ⏳ Planned |
| Phase 1b | ⏳ Planned |
| Phase 1c | ⏳ Planned |
| Downstream classifier | ⏳ Future |
| Clinical evaluation | ❌ Not performed |

---

## What This Project Is — and Isn't

### It is:

- 🧑‍🎓 A student project
- 🔬 A proof of concept
- 🧬 An exploration of genomic language modeling
- 💻 An experiment using relatively accessible hardware
- 🇲🇦 A project motivated by North African genomic representation
- 📚 An opportunity to learn about genomics, Transformers, population genetics, and machine learning

### It is not:

- ❌ A clinical diagnostic tool
- ❌ A medically validated AI system
- ❌ A replacement for geneticists or clinical laboratories
- ❌ A production-ready VUS classifier
- ❌ Evidence that North African patients currently receive better variant interpretation
- ❌ A claim that the proposed approach will outperform existing methods

**The project is an experiment. The results are not known yet.**

---

## Why Build It?

The primary goal of AtlasGen-SLM is not to immediately produce a clinically deployable system.

It is to answer a much smaller question:

> **Can a medical student, using a relatively modest GPU and publicly available genomic resources, build and train a small genomic language model that learns meaningful representations of human genetic variation?**

If the answer is yes, the project can then serve as a foundation for more serious experiments.

If the answer is no, that is also a useful result.

---

## Author

**Bassam**  
Medical student — Faculty of Medicine and Pharmacy of Casablanca (FMPC), Morocco.

AtlasGen-SLM is an independent student research project exploring the intersection of:

- genomics
- artificial intelligence
- population genetics
- medical research

---

## Disclaimer

> **AtlasGen-SLM is a student-built proof of concept and should be treated as an experimental research project.**
>
> The model has not yet been trained, clinically validated, or evaluated for diagnostic use. Any future results should be interpreted as research findings rather than medical evidence or clinical recommendations.

---

## References

The project draws on publicly available genomic resources including:

- [1000 Genomes Project](https://www.internationalgenome.org/)
- [Human Genome Diversity Project (HGDP)](https://www.internationalgenome.org/data-portal/data-collection/hgdp/)
- [Simons Genome Diversity Project (SGDP)](https://www.simonsfoundation.org/simons-genome-diversity-project/)
- [gnomAD](https://gnomad.broadinstitute.org/)
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)

Dataset provenance, processing decisions, and experimental details will be documented as the project develops.
