# AtlasGen-SLM

> A ~15M parameter genomic language model for variant pathogenicity classification in North African and Moroccan populations.

---

## The Problem

Genomic AI tools are trained almost exclusively on Western European data. When they encounter a variant they've never seen — which happens constantly with North African genomes — they return "Variant of Uncertain Significance." Not because the variant is actually uncertain. Because the training data never included it.

That lands as a VUS — a result that's technically inconclusive — for Moroccan and Maghrebi patients at a much higher rate than for European ones. Resolving a VUS means more testing, more time, more cost. In Morocco's healthcare system, those aren't abstract problems.

---

## What AtlasGen-SLM Does

It's a variant pathogenicity classifier. Feed it a genomic variant, get back a probability: benign, pathogenic, or uncertain. The difference from existing tools is the training data — built around African, Maghrebi, and Moroccan genomes rather than European ones.

It runs offline. The trained weights sit on disk. No API, no cloud, no per-query cost. A clinic with no stable internet can run it.

---

## How It Works

```
Input: variant (chr, pos, ref, alt)
    ↓
Extract 1024bp window from GRCh38 around the variant position
Inject the alternate allele at the variant site
    ↓
Tokenize into overlapping 6-mers (vocab: 4,096 tokens)
    ↓
Pass through encoder-only transformer (~15M parameters)
    ↓
Output: Benign / Pathogenic / Uncertain + confidence score
```

Training runs in four sequential phases — global pretraining, African and Maghrebi fine-tuning, Moroccan specialization, then supervised pathogenicity classification. Details in [`docs/architecture.md`](docs/architecture.md).

---

## Model Specs

| | |
|---|---|
| Architecture | Encoder-only transformer |
| Parameters | ~15M |
| Tokenization | 6-mer sliding window, vocab 4,096 |
| Sequence input | 1024bp VCF-guided flanks |
| Training hardware | 8GB VRAM |
| Attention | FlashAttention |
| Optimizer | PagedAdamW (fp16) |
| Inference | Fully offline |

---

## Input / Output

**Input**
```
Chromosome: 11 | Position: 5246696 | Ref: A | Alt: T
```

**Output**
```
Variant:         HBB c.20A>T
Classification:  Pathogenic
Confidence:      0.94
Population note: Elevated frequency in North African cohorts — regional context applied
```

**On zygosity:** The model evaluates sequence-level pathogenic propensity — it does not account for whether a variant is heterozygous or homozygous. Zygosity is a clinical interpretation layer that sits above the model output. A pathogenic call means the variant's sequence context is consistent with disease; whether one or two copies produces that disease depends on the gene's inheritance pattern and is handled downstream.

---

## Training Data

### Phase 1 — Global Baseline

| Dataset | Samples | Access |
|---|---|---|
| 1000 Genomes Project | 2,504 | Open |
| Simons Genome Diversity Project (SGDP) | 300 WGS | Open (ENA: PRJEB9586) |
| Human Genome Diversity Project (HGDP) | 929 WGS | Open via IGSR ⚠️ not AnVIL |
| gnomAD | 125k exomes / 15k WGS | Open (aggregated) |
| TOPMed | 150,000+ WGS | Controlled (dbGaP) |

### Phase 2 — African & Maghrebi

| Dataset | Samples | Access |
|---|---|---|
| H3Africa WGS (EGAS00001005972) | 426 WGS | Controlled (H3Africa DAC) |
| Egypt Genome Project 1K (EGP1K) | 1,024 WGS | Controlled (ECRRM) |
| GME Variome — Northwest Africa cohort | 85 exomes | Controlled (contact PIs) |
| Saudi Genome Program v1 | 1,378 WGS | Controlled |

### Phase 3 — Moroccan

| Dataset | Samples | Access |
|---|---|---|
| Moroccan Genome Project (EGAD50000000750) | 109 WGS | Controlled (EGA DAC) |
| Tunisia & Morocco array (EGAD00001009071) | 109 samples | Controlled |
| Algerian Amazigh array (EGAD00001010900) | 164 samples | Controlled |

### Phase 4 — Pathogenic Variant Databases

| Dataset | Variants | Access |
|---|---|---|
| ClinVar | ~1M | Open |
| HGMD | ~400k | Commercial |
| LOVD | 4.6M | Open (API) |
| Mastermind | 6.1M | Free academic tier |

Full dataset provenance, access procedures, and curation notes are in [`docs/datasets.md`](docs/datasets.md).

**Access fallback:** Most Phase 2 and 3 datasets require DAC approval, which can take months. If approvals are delayed, Phase 1 and early Phase 2 training will proceed on open-access data that already includes North African samples — SGDP and HGDP both contain Mozabite genomes, and the 1000 Genomes African cohort provides continental-level signal. These are not substitutes for the controlled datasets, but they're enough to develop and validate the pipeline while applications process.

---

## Validation

Benchmarked on HBB (beta-globin) and G6PD variants — two genes with well-characterized North African allele frequencies where Western tools are known to produce high VUS rates. Primary metric: the proportion of variants existing tools call uncertain that AtlasGen-SLM resolves to a confident call.

---

## Context

When Moroccan patients get whole exome sequencing today, the analysis typically runs through Western commercial pipelines. Those pipelines weren't built for this population. The VUS rate climbs, the diagnostic timeline stretches, and the cost lands on the patient.

In April 2026, Morocco's FM6SS opened a Precision Medicine Hub in Rabat. The sequencing infrastructure is coming. A classifier trained on Moroccan data should have been here already.

---

## Project Status

| | |
|---|---|
| Repository setup | 🔄 In progress |
| Data curation | 🔄 In progress |
| Tokenizer | ⏳ Planned |
| Model architecture | ⏳ Planned |
| Training — all phases | ⏳ Planned |
| Evaluation on HBB / G6PD | ⏳ Planned |

---

## Author

**Bassam**
First-year medical student, Faculty of Medicine and Pharmacy of Casablanca (FMPC)

---

## References

- Moroccan Genome Project: [EGA EGAD50000000750](https://ega-archive.org/datasets/EGAD50000000750)
- SGDP: [PMC5161557](https://pmc.ncbi.nlm.nih.gov/articles/PMC5161557/)
- H3Africa: [h3africa.org](https://h3africa.org)
- EGP1K: [bioRxiv 2026.04.02.715521](https://www.biorxiv.org/content/10.64898/2026.04.02.715521v1)
- FM6SS Hub: [Morocco World News, April 2026](https://www.moroccoworldnews.com/2026/04/286002/fm6ss-launches-advanced-hub-to-integrate-genomics-patient-care-in-rabat/)
